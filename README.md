# DeepPerson

[![PyPI version](https://badge.fury.io/py/deep-person.svg)](https://badge.fury.io/py/deep-person)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple person re-identification system with automatic person detection and embedding generation.

## Installation

```bash
pip install deep-person
```

## Quick Start

```python
from deep_person import DeepPerson

dp = DeepPerson()

# Generate embeddings
result = dp.represent("person.jpg")

# Verify if two images show the same person
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']}")
```

## Features

- **Automatic person detection** - YOLO-based detection
- **Multi-modal embeddings** - Body (2048-dim) + Face (512-dim) or via Model Registry
- **Identity verification** - Cosine/Euclidean distance metrics
- **GPU acceleration** - Automatic CUDA detection with CPU fallback
- **Batch processing** - Efficient multi-image handling

## Requirements

- Python 3.11+
- Models downloaded automatically on first use
- Optional: CUDA GPU for faster processing
