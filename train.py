"""
Model training script for Multiclass Fish Image Classification.
"""

import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import load_dataset, build_transfer_model


def build_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Build a simple CNN from scratch for fish classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of fish species classes
    
    Returns:
        model: Compiled Keras Sequential model
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Fourth Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def main():
    """Main training function."""
    print("=" * 60)
    print("Multiclass Fish Image Classification - CNN Training")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    train_gen, val_gen, num_classes = load_dataset(data_dir='./data')
    
    # Build model
    print(f"\n[2/4] Building CNN model for {num_classes} classes...")
    model = build_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
    
    # Compile model
    print("\n[3/4] Compiling model...")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Setup callbacks
    os.makedirs('./models', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath='./models/cnn_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train model
    print("\n[4/4] Training model...")
    print("-" * 60)
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[checkpoint_callback],
        verbose=1
    )
    
    # Print training history keys
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print("\nTraining history keys:", list(history.history.keys()))
    
    return model, history


def train_transfer_model(base_model_name, epochs=10, data_dir='./data'):
    """
    Train a transfer learning model using a pre-trained architecture.
    
    Args:
        base_model_name (str): Name of the base model. Supported:
                              'VGG16', 'ResNet50', 'MobileNet', 'InceptionV3', 'EfficientNetB0'
        epochs (int): Number of training epochs. Default: 10
        data_dir (str): Path to dataset directory. Default: './data'
    
    Returns:
        tuple: (model, history)
    """
    print("=" * 60)
    print(f"Transfer Learning Training - {base_model_name}")
    print("=" * 60)
    
    # Determine input shape and target size based on model
    if base_model_name == 'InceptionV3':
        input_shape = (299, 299, 3)
        target_size = (299, 299)
        print(f"\nNote: {base_model_name} uses input size (299, 299)")
    else:
        input_shape = (224, 224, 3)
        target_size = (224, 224)
    
    # Load dataset with appropriate target size
    print("\n[1/4] Loading dataset...")
    train_gen, val_gen, num_classes = load_dataset(
        data_dir=data_dir,
        target_size=target_size
    )
    
    # Build transfer learning model
    print(f"\n[2/4] Building {base_model_name} transfer learning model...")
    model = build_transfer_model(
        base_model_name=base_model_name,
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Setup callbacks
    os.makedirs('./models', exist_ok=True)
    model_filename = f"./models/{base_model_name.lower()}_best.h5"
    checkpoint_callback = ModelCheckpoint(
        filepath=model_filename,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train model
    print(f"\n[3/4] Training {base_model_name} model...")
    print("-" * 60)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint_callback],
        verbose=1
    )
    
    # Print final validation accuracy
    final_val_accuracy = max(history.history['val_accuracy'])
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nBest validation accuracy: {final_val_accuracy:.4f}")
    print(f"Model saved to: {model_filename}")
    print(f"\nTraining history keys: {list(history.history.keys())}")
    
    return model, history, final_val_accuracy


def train_multiple_models(model_names, epochs=10, data_dir='./data'):
    """
    Train multiple transfer learning models and compare results.
    
    Args:
        model_names (list): List of model names to train
        epochs (int): Number of training epochs for each model
        data_dir (str): Path to dataset directory
    
    Returns:
        dict: Results dictionary with model names and their best validation accuracies
    """
    valid_models = ["VGG16", "ResNet50", "MobileNet", "InceptionV3", "EfficientNetB0"]
    
    # Validate model names
    invalid_models = [m for m in model_names if m not in valid_models]
    if invalid_models:
        raise ValueError(
            f"Invalid model names: {invalid_models}. "
            f"Valid models: {valid_models}"
        )
    
    results = {}
    
    print("\n" + "=" * 60)
    print(f"Training {len(model_names)} Transfer Learning Model(s)")
    print("=" * 60)
    print(f"Models to train: {', '.join(model_names)}")
    print(f"Epochs per model: {epochs}")
    print("=" * 60)
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"Training Model {i}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        try:
            _, history, best_val_acc = train_transfer_model(
                base_model_name=model_name,
                epochs=epochs,
                data_dir=data_dir
            )
            results[model_name] = {
                'best_val_accuracy': float(best_val_acc),
                'epochs': epochs,
                'status': 'completed'
            }
            print(f"\n✓ {model_name} training completed successfully!")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
        except Exception as e:
            print(f"\n✗ {model_name} training failed: {str(e)}")
            results[model_name] = {
                'best_val_accuracy': None,
                'epochs': epochs,
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def print_comparison_table(results):
    """Print a comparison table of model results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Sort results by accuracy (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['best_val_accuracy'] if x[1]['best_val_accuracy'] is not None else -1,
        reverse=True
    )
    
    print(f"\n{'Model Name':<20} {'Best Val Accuracy':<20} {'Status':<15}")
    print("-" * 60)
    
    for model_name, result in sorted_results:
        if result['best_val_accuracy'] is not None:
            acc_str = f"{result['best_val_accuracy']:.4f}"
        else:
            acc_str = "N/A"
        status = result['status']
        print(f"{model_name:<20} {acc_str:<20} {status:<15}")
    
    print("\n" + "=" * 60)


def save_results_json(results, filepath='./reports/transfer_results.json'):
    """Save results to JSON file."""
    os.makedirs('./reports', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    # Valid model names
    valid_models = ["VGG16", "ResNet50", "MobileNet", "InceptionV3", "EfficientNetB0"]
    
    print("=" * 60)
    print("Multiclass Fish Image Classification - Training")
    print("=" * 60)
    print("\nAvailable models:")
    for i, model in enumerate(valid_models, 1):
        print(f"  {i}. {model}")
    
    print("\n" + "-" * 60)
    print("Training Options:")
    print("  1. Train CNN from scratch")
    print("  2. Train transfer learning model(s)")
    print("-" * 60)
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Train CNN from scratch
        model, history = main()
    elif choice == "2":
        # Train transfer learning models
        print("\nEnter model names to train (comma-separated, e.g., VGG16,ResNet50)")
        print("Or enter 'all' to train all models")
        user_input = input("Model names: ").strip()
        
        if user_input.lower() == 'all':
            model_names = valid_models
        else:
            model_names = [m.strip() for m in user_input.split(',')]
            # Validate
            invalid = [m for m in model_names if m not in valid_models]
            if invalid:
                print(f"Error: Invalid model names: {invalid}")
                exit(1)
        
        epochs_input = input("Number of epochs per model (default: 10): ").strip()
        epochs = int(epochs_input) if epochs_input else 10
        
        # Train multiple models
        results = train_multiple_models(model_names, epochs=epochs)
        
        # Print comparison table
        print_comparison_table(results)
        
        # Save results to JSON
        save_results_json(results)
    else:
        print("Invalid choice. Exiting.")
