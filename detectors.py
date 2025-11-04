"""
Person detection abstraction for DeepPerson library.

Provides detector backends for locating persons in images:
- UltralyticsDetector: YOLO-based detection (default, recommended)
- TorchVisionDetector: Faster R-CNN fallback
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DetectionResult:
    """
    Result from person detection.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in absolute pixels
        confidence: Detection confidence score [0, 1]
        class_name: Detected class name (should be 'person')
        class_id: Detected class ID
    """

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        confidence: float,
        class_name: str = "person",
        class_id: int = 0
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id

    def __repr__(self) -> str:
        return (
            f"DetectionResult(bbox={self.bbox}, confidence={self.confidence:.3f}, "
            f"class='{self.class_name}')"
        )


class PersonDetector(ABC):
    """
    Abstract base class for person detectors.

    Subclasses must implement detect() and crop_persons().
    """

    @abstractmethod
    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Detect persons in an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of DetectionResult objects
        """
        pass

    @abstractmethod
    def crop_persons(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        detections: List[DetectionResult]
    ) -> List[Image.Image]:
        """
        Crop detected persons from image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            detections: List of DetectionResult objects

        Returns:
            List of cropped PIL Image objects
        """
        pass

    @staticmethod
    def _load_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various input formats.

        Args:
            image: Input in various formats

        Returns:
            PIL Image object
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


class UltralyticsDetector(PersonDetector):
    """
    Person detection using Ultralytics YOLO models.

    Default and recommended detector backend. Uses YOLOv8 pretrained on COCO.
    Supports both CPU and GPU inference with good performance.

    References:
        https://docs.ultralytics.com/usage/python/
        https://docs.ultralytics.com/tasks/detect/
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "yolov8n.pt"
    ):
        """
        Initialize Ultralytics YOLO detector.

        Args:
            device: torch.device for inference
            model_name: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        self.device = device
        self.model_name = model_name

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required for UltralyticsDetector. "
                "Install with: pip install ultralytics"
            )

        # Ensure model weights are available using model manager
        from .model_manager import get_model_manager
        model_manager = get_model_manager()
        weights_path = model_manager.ensure_yolo_weights(model_name)

        # Load YOLO model with managed weights
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(str(weights_path))

        # Move to device
        if device.type == "cuda":
            # Ultralytics handles device internally
            self.device_str = f"cuda:{device.index or 0}"
        else:
            self.device_str = "cpu"

        logger.info(f"UltralyticsDetector initialized on {self.device_str}")

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Detect persons in image using YOLO.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of DetectionResult objects for person class only
        """
        # Load image
        img = self._load_image(image)

        # Run YOLO inference
        results = self.model.predict(
            source=img,
            conf=confidence_threshold,
            device=self.device_str,
            verbose=False
        )

        # Extract person detections (class 0 in COCO)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detection is 'person' class (class_id = 0 in COCO)
                class_id = int(box.cls[0])
                if class_id == 0:  # person class
                    # Get bbox coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])

                    detection = DetectionResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_name="person",
                        class_id=class_id
                    )
                    detections.append(detection)

        logger.debug(f"Detected {len(detections)} person(s) with YOLO")
        return detections

    def crop_persons(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        detections: List[DetectionResult]
    ) -> List[Image.Image]:
        """
        Crop detected persons from image.

        Args:
            image: Input image
            detections: List of DetectionResult objects

        Returns:
            List of cropped PIL Images
        """
        img = self._load_image(image)
        crops = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            # Crop the region
            cropped = img.crop((x1, y1, x2, y2))
            crops.append(cropped)

        return crops


class TorchVisionDetector(PersonDetector):
    """
    Person detection using TorchVision Faster R-CNN.

    Fallback detector using pretrained Faster R-CNN with ResNet-50 backbone.
    Trained on COCO dataset.
    """

    def __init__(self, device: torch.device):
        """
        Initialize TorchVision Faster R-CNN detector.

        Args:
            device: torch.device for inference
        """
        self.device = device

        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        except ImportError:
            raise ImportError(
                "torchvision required for TorchVisionDetector. "
                "Install with: pip install torchvision"
            )

        # Load pretrained Faster R-CNN
        logger.info("Loading Faster R-CNN model")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(device)
        self.model.eval()

        # COCO class names (person is class 1)
        self.coco_person_class_id = 1

        logger.info(f"TorchVisionDetector initialized on {device}")

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Detect persons using Faster R-CNN.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of DetectionResult objects for person class only
        """
        import torchvision.transforms as T

        # Load and prepare image
        img = self._load_image(image)

        # Convert to tensor
        transform = T.ToTensor()
        img_tensor = transform(img).to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        # Extract person detections
        detections = []
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            # Check if detection is person and above threshold
            if label == self.coco_person_class_id and score >= confidence_threshold:
                x1, y1, x2, y2 = box
                detection = DetectionResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(score),
                    class_name="person",
                    class_id=int(label)
                )
                detections.append(detection)

        logger.debug(f"Detected {len(detections)} person(s) with Faster R-CNN")
        return detections

    def crop_persons(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        detections: List[DetectionResult]
    ) -> List[Image.Image]:
        """
        Crop detected persons from image.

        Args:
            image: Input image
            detections: List of DetectionResult objects

        Returns:
            List of cropped PIL Images
        """
        img = self._load_image(image)
        crops = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            # Crop the region
            cropped = img.crop((x1, y1, x2, y2))
            crops.append(cropped)

        return crops


class DetectorFactory:
    """
    Factory for creating person detector instances.
    """

    _BACKENDS = {
        "yolo": UltralyticsDetector,
        "ultralytics": UltralyticsDetector,
        "fasterrcnn": TorchVisionDetector,
        "torchvision": TorchVisionDetector,
    }

    @classmethod
    def create_detector(
        cls,
        backend: str = "yolo",
        device: Optional[torch.device] = None,
        **kwargs
    ) -> PersonDetector:
        """
        Create a person detector instance.

        Args:
            backend: Detector backend ('yolo', 'ultralytics', 'fasterrcnn', 'torchvision')
            device: torch.device for inference (auto-detected if None)
            **kwargs: Additional arguments for detector initialization

        Returns:
            PersonDetector instance

        Raises:
            ValueError: If backend not supported
        """
        backend_lower = backend.lower()

        if backend_lower not in cls._BACKENDS:
            available = ", ".join(cls._BACKENDS.keys())
            raise ValueError(
                f"Unsupported detector backend: '{backend}'. "
                f"Available: {available}"
            )

        # Auto-detect device if not provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Auto-detected device: {device}")

        # Create detector
        detector_class = cls._BACKENDS[backend_lower]
        return detector_class(device=device, **kwargs)

    @classmethod
    def list_available_backends(cls) -> List[str]:
        """
        List available detector backend names.

        Returns:
            List of backend identifiers
        """
        return list(cls._BACKENDS.keys())


def get_default_detector(device: Optional[torch.device] = None) -> PersonDetector:
    """
    Get default person detector (YOLO-based).

    Args:
        device: torch.device for inference (auto-detected if None)

    Returns:
        Default PersonDetector instance (UltralyticsDetector)
    """
    return DetectorFactory.create_detector(backend="yolo", device=device)
