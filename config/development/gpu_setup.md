# Development Environment Configuration for User Gallery Fusion

This document provides configuration guidance for setting up the development environment with GPU support for face embeddings.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 30xx, 40xx series recommended)
- **VRAM**: Minimum 8GB VRAM for face embedding models
- **CUDA**: CUDA 11.8 or later for optimal performance
- **Driver**: Latest NVIDIA drivers

### Software Requirements
- **Python**: 3.12+
- **PyTorch**: 2.0+ with CUDA support
- **CUDA Toolkit**: 11.8+
- **cuDNN**: 8.6+

## Environment Setup

### 1. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Install DeepFace with GPU Support

```bash
# Install DeepFace
pip install deepface>=0.0.82

# Install additional dependencies for GPU support
pip install tensorflow-gpu>=2.13.0  # Required for some DeepFace backends
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="8.6"

# DeepFace Configuration
DEEPFACE_BACKEND=tensorflow
DEEPFACE_MODEL_NAME=Facenet
DEEPFACE_DETECTOR_BACKEND=opencv

# Performance Configuration
DEEPFACE_ENABLE_GPU=True
DEEPFACE_BATCH_SIZE=32
DEEPFACE_FORCE_DOWNLOAD=False
```

### 4. Verify GPU Setup

Create a verification script:

```python
import torch
import os

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name()}")

# Check DeepFace GPU support
try:
    from deepface import DeepFace
    print(f"DeepFace version: {DeepFace.__version__}")
    print("DeepFace installed successfully")
except ImportError as e:
    print(f"DeepFace import error: {e}")

# Test GPU memory
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

## Configuration Files

### PyTorch Configuration (`torch_config.py`)

```python
import torch
import os

class TorchConfig:
    """PyTorch configuration for optimal performance."""

    def __init__(self):
        self.device = self._get_device()
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.num_workers = min(4, os.cpu_count() or 1)
        self.pin_memory = self.device.type == 'cuda'

    def _get_device(self) -> torch.device:
        """Get the optimal device for computation."""
        if torch.cuda.is_available():
            device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
            return torch.device(f'cuda:{device_id}')
        return torch.device('cpu')

    def get_torch_config(self) -> dict:
        """Get PyTorch configuration dictionary."""
        return {
            'device': self.device,
            'dtype': self.dtype,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
```

### DeepFace Configuration (`deepface_config.py`)

```python
import os
from typing import Dict, Any

class DeepFaceConfig:
    """DeepFace configuration for optimal face embedding generation."""

    def __init__(self):
        self.model_name = os.environ.get('DEEPFACE_MODEL_NAME', 'Facenet')
        self.detector_backend = os.environ.get('DEEPFACE_DETECTOR_BACKEND', 'opencv')
        self.enable_gpu = os.environ.get('DEEPFACE_ENABLE_GPU', 'True').lower() == 'true'
        self.batch_size = int(os.environ.get('DEEPFACE_BATCH_SIZE', '32'))
        self.force_download = os.environ.get('DEEPFACE_FORCE_DOWNLOAD', 'False').lower() == 'true'

    def get_config(self) -> Dict[str, Any]:
        """Get DeepFace configuration dictionary."""
        return {
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'enable_gpu': self.enable_gpu,
            'batch_size': self.batch_size,
            'force_download': self.force_download
        }
```

## Performance Optimization

### 1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision for faster computation
scaler = GradScaler()

with autocast():
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Memory Optimization

```python
# Clear GPU cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Optimize batch size based on available memory
def get_optimal_batch_size():
    if torch.cuda.is_available():
        # Estimate based on GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Reserve memory for model and system
        available_memory = total_memory * 0.7
        # Estimate batch size (adjust based on model size)
        return max(1, int(available_memory / (1024 * 1024 * 100)))  # Rough estimate
    return 32  # Default CPU batch size
```

### 3. Data Loading Optimization

```python
from torch.utils.data import DataLoader

def create_optimized_dataloader(dataset, batch_size, num_workers=None, pin_memory=None):
    """Create an optimized DataLoader for GPU training."""
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **DeepFace Import Errors**
   - Ensure TensorFlow GPU is installed
   - Check CUDA and cuDNN compatibility
   - Verify environment variables

3. **Slow Performance**
   - Enable mixed precision
   - Optimize data loading
   - Check GPU utilization

### Verification Commands

```bash
# Check GPU status
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Test DeepFace
python -c "from deepface import DeepFace; print(DeepFace.__version__)"

# Run performance test
python -m pytest tests/integration/test_performance.py -v
```

## Development Environment Setup Script

Create a setup script (`setup_dev_env.py`):

```python
#!/usr/bin/env python3
"""
Development environment setup script for User Gallery Fusion.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_cuda():
    """Check CUDA availability and version."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available (version: {torch.version.cuda})")
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
            return True
        else:
            print("❌ CUDA is not available")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'deepface'])

def create_env_file():
    """Create .env file with default configuration."""
    env_content = """
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="8.6"

# DeepFace Configuration
DEEPFACE_BACKEND=tensorflow
DEEPFACE_MODEL_NAME=Facenet
DEEPFACE_DETECTOR_BACKEND=opencv

# Performance Configuration
DEEPFACE_ENABLE_GPU=True
DEEPFACE_BATCH_SIZE=32
DEEPFACE_FORCE_DOWNLOAD=False
"""
    with open('.env', 'w') as f:
        f.write(env_content.strip())
    print("✅ Created .env file")

def main():
    """Main setup function."""
    print("Setting up development environment for User Gallery Fusion...")

    # Check CUDA
    cuda_available = check_cuda()

    # Install dependencies
    install_dependencies()

    # Create environment file
    create_env_file()

    print("\nSetup complete!")
    if cuda_available:
        print("🎉 GPU acceleration is available")
    else:
        print("⚠️  GPU acceleration is not available - using CPU")

if __name__ == "__main__":
    main()
```

Run the setup script:
```bash
python setup_dev_env.py
```

This configuration provides optimal GPU support for face embedding generation while maintaining compatibility with the existing DeepPerson framework.