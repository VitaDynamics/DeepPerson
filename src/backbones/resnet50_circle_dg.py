"""
ResNet-50 with Circle DG Loss Backbone

Implements the layumi/Person_reID_baseline_pytorch ft_net architecture
for person re-identification with Circle loss and Domain Generalization.

Reference:
    https://github.com/layumi/Person_reID_baseline_pytorch
    Model: ResNet-50 (all tricks + Circle + DG)
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


def weights_init_kaiming(m):
    """Kaiming initialization for linear and conv layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class ClassBlock(nn.Module):
    """
    Classification/Feature extraction block for person re-ID.

    This is the 'neck' architecture from layumi's baseline that sits
    on top of the ResNet backbone.
    """

    def __init__(
        self,
        input_dim: int,
        class_num: int,
        droprate: float = 0.5,
        relu: bool = False,
        bnorm: bool = True,
        linear: int = 512,
        return_f: bool = False
    ):
        """
        Initialize ClassBlock.

        Args:
            input_dim: Input feature dimension (2048 for ResNet-50)
            class_num: Number of person identities (for training)
            droprate: Dropout rate
            relu: Whether to use LeakyReLU
            bnorm: Whether to use BatchNorm
            linear: Linear projection dimension (0 means no projection)
            return_f: Whether to return features (True for inference)
        """
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        # Feature transformation block
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim

        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        # Classifier (not used for inference)
        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)

        self.linear_num = linear
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        """Forward pass through ClassBlock."""
        x = self.add_block(x)
        if self.return_f:
            # Return features only (for inference)
            return x
        else:
            # Return classification output (for training)
            x = self.classifier(x)
            return x


class FtNet(nn.Module):
    """
    ft_net: ResNet-50 backbone with person re-ID adaptations.

    Based on layumi/Person_reID_baseline_pytorch model.py
    Supports Circle loss, IBN (Instance-Batch Normalization), and stride modifications.
    """

    def __init__(
        self,
        class_num: int = 751,
        droprate: float = 0.5,
        stride: int = 2,
        circle: bool = False,
        ibn: bool = False,
        linear_num: int = 512
    ):
        """
        Initialize ft_net.

        Args:
            class_num: Number of person identities (used during training)
            droprate: Dropout rate
            stride: Stride for layer4 (1 or 2)
            circle: Whether Circle loss was used (return features if True)
            ibn: Whether to use IBN-Net variant
            linear_num: Feature dimension after linear projection (0 for 2048-dim)
        """
        super(FtNet, self).__init__()

        # Load ResNet-50 backbone
        if ibn:
            # IBN-Net variant (requires torch.hub access)
            try:
                model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=False)
                logger.info("Using IBN-Net ResNet-50 variant")
            except Exception as e:
                logger.warning(f"Failed to load IBN-Net, falling back to standard ResNet-50: {e}")
                model_ft = models.resnet50(pretrained=False)
        else:
            model_ft = models.resnet50(pretrained=False)

        # Modify stride if requested (for higher resolution features)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        # Use adaptive average pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(
            2048,
            class_num,
            droprate,
            linear=linear_num,
            return_f=circle  # Return features for Circle loss / inference
        )

    def forward(self, x):
        """Forward pass through ft_net."""
        # ResNet-50 backbone
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global average pooling
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classification/Feature block
        x = self.classifier(x)
        return x


def load_model(
    weights_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    feature_dim: int = 512,
    stride: int = 2,
    ibn: bool = False,
    use_model_manager: bool = True
) -> nn.Module:
    """
    Load ft_net model for inference.

    Args:
        weights_path: Path to pretrained .pth weights file (if None, uses model_manager)
        device: Target device (cuda or cpu)
        feature_dim: Output feature dimension (512 or 2048)
            512: Uses linear projection (typical for Circle loss training)
            2048: Direct ResNet-50 features (no projection)
        stride: Layer4 stride (1 for high-res features, 2 for standard)
        ibn: Whether the weights use IBN-Net architecture
        use_model_manager: Whether to use model_manager for automatic download

    Returns:
        Loaded PyTorch model in eval mode

    Raises:
        FileNotFoundError: If weights file not found
        RuntimeError: If weights loading fails
    """
    # Use model manager to ensure weights are available
    if use_model_manager and weights_path is None:
        try:
            from ..model_manager import get_model_manager
            model_manager = get_model_manager()
            weights_path = model_manager.ensure_backbone_weights("resnet50_circle_dg")


        except ImportError:
            logger.warning("model_manager not available, falling back to manual weights_path")
            if weights_path is None:
                raise ValueError("weights_path must be provided when model_manager is not available")
    else:
        logger.info(f"Loading ft_net from {weights_path}")

    # Determine linear_num from feature_dim
    if feature_dim == 512:
        linear_num = 512
    elif feature_dim == 2048:
        linear_num = 0  # No linear projection
    else:
        raise ValueError(f"feature_dim must be 512 or 2048, got {feature_dim}")

    # Create model with circle=True for inference (return features only)
    model = FtNet(
        class_num=751,  # Doesn't matter for inference, but needed for weight loading
        droprate=0.5,
        stride=stride,
        circle=True,  # Return features only
        ibn=ibn,
        linear_num=linear_num
    )

    # Load weights
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Please download from: https://github.com/layumi/Person_reID_baseline_pytorch"
        )

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {weights_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}, feature_dim={feature_dim}")

    return model


class ResNet50CircleDGWrapper:
    """
    Convenience wrapper for ft_net inference.

    Provides a simple interface for embedding extraction.
    """

    def __init__(self, model: nn.Module, device: torch.device, feature_dim: int):
        """
        Initialize wrapper.

        Args:
            model: Loaded ft_net model
            device: Device model is on
            feature_dim: Feature dimension
        """
        self.model = model
        self.device = device
        self.feature_dim = feature_dim

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Feature tensor (B, feature_dim)
        """
        self.model.eval()
        features = self.model(x)
        return features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow calling wrapper as a function."""
        return self.extract_features(x)
