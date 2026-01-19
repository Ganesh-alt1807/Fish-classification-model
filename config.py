"""
Configuration file for Fish Classification Model
"""

import os

# Dataset configuration
# The actual dataset is in ./Dataset/data/
# But the code expects ./data/
# We'll use the Dataset/data path as default

DATA_DIR = './Dataset/data'  # Main dataset location

# Model configuration
MODELS_DIR = './models'
REPORTS_DIR = './reports'

# Training configuration
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2

# Image preprocessing
DEFAULT_IMAGE_SIZE = (224, 224)
INCEPTION_IMAGE_SIZE = (299, 299)

# Supported models
SUPPORTED_TRANSFER_MODELS = [
    'VGG16',
    'ResNet50',
    'MobileNet',
    'InceptionV3',
    'EfficientNetB0'
]

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Dataset directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Reports directory: {REPORTS_DIR}")
