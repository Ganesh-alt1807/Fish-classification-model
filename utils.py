"""
Helper functions for Multiclass Fish Image Classification.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_class_labels(data_dir='./data'):
    """
    Get class labels from dataset directory structure.
    
    Args:
        data_dir (str): Path to dataset directory containing class subfolders
    
    Returns:
        list: Sorted list of class label names
    """
    if not os.path.exists(data_dir):
        return []
    
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    return sorted(class_dirs)


def detect_model_type(model_path):
    """
    Detect model type from filename.
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        tuple: (model_name, target_size)
    """
    filename = os.path.basename(model_path).lower()
    
    if 'cnn' in filename:
        model_name = 'CNN'
        target_size = (224, 224)
    elif 'inceptionv3' in filename or 'inception' in filename:
        model_name = 'InceptionV3'
        target_size = (299, 299)
    elif 'vgg16' in filename:
        model_name = 'VGG16'
        target_size = (224, 224)
    elif 'resnet50' in filename or 'resnet' in filename:
        model_name = 'ResNet50'
        target_size = (224, 224)
    elif 'mobilenet' in filename:
        model_name = 'MobileNet'
        target_size = (224, 224)
    elif 'efficientnet' in filename or 'efficientnetb0' in filename:
        model_name = 'EfficientNetB0'
        target_size = (224, 224)
    else:
        # Default: assume 224x224 for unknown models
        model_name = 'Unknown'
        target_size = (224, 224)
    
    return model_name, target_size


def load_dataset(data_dir='./data', target_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load dataset using ImageDataGenerator with automatic train/validation split.
    
    Args:
        data_dir (str): Path to the dataset directory containing class subfolders.
                       Default: './data'
        target_size (tuple): Target image size (height, width). Default: (224, 224)
        batch_size (int): Batch size for data generators. Default: 32
        validation_split (float): Fraction of data to use for validation. Default: 0.2
    
    Returns:
        tuple: (train_generator, validation_generator, num_classes)
            - train_generator: ImageDataGenerator for training data
            - validation_generator: ImageDataGenerator for validation data
            - num_classes (int): Number of fish species classes
    """
    # Count number of classes by counting subdirectories in data_dir
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    num_classes = len(class_dirs)
    
    print(f"Found {num_classes} classes in dataset")
    print(f"Classes: {', '.join(sorted(class_dirs))}")
    
    # Create ImageDataGenerator with data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Create ImageDataGenerator for validation (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create training generator
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Create validation generator
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {target_size}")
    
    return train_gen, val_gen, num_classes


def build_transfer_model(base_model_name, input_shape, num_classes):
    """
    Build a transfer learning model using a pre-trained base architecture.
    
    Args:
        base_model_name (str): Name of the base model. Supported: 
                              'VGG16', 'ResNet50', 'MobileNet', 'InceptionV3', 'EfficientNetB0'
        input_shape (tuple): Input shape (height, width, channels)
        num_classes (int): Number of output classes
    
    Returns:
        model: Compiled Keras Model with transfer learning architecture
    """
    # Map model names to their Keras applications
    base_models = {
        'VGG16': VGG16,
        'ResNet50': ResNet50,
        'MobileNet': MobileNet,
        'InceptionV3': InceptionV3,
        'EfficientNetB0': EfficientNetB0
    }
    
    if base_model_name not in base_models:
        raise ValueError(
            f"Unsupported base model: {base_model_name}. "
            f"Supported models: {list(base_models.keys())}"
        )
    
    print(f"\nBuilding transfer learning model with {base_model_name}...")
    
    # Load base model with ImageNet weights
    BaseModel = base_models[base_model_name]
    
    # Handle different input requirements
    if base_model_name == 'InceptionV3':
        # InceptionV3 requires 299x299 input
        if input_shape[:2] != (299, 299):
            print(f"Warning: {base_model_name} typically uses (299, 299) input size.")
            print(f"Using provided input_shape: {input_shape}")
        base_model = BaseModel(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        base_model = BaseModel(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    
    # Freeze base model layers
    base_model.trainable = False
    print(f"Base model '{base_model_name}' loaded with ImageNet weights (frozen)")
    
    # Build the complete model
    inputs = base_model.input
    x = base_model.output
    
    # Add custom classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    # Count trainable parameters
    trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
    
    print(f"Transfer learning model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
