"""
Model evaluation script for Multiclass Fish Image Classification.
Generates metrics and visualization plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
from utils import load_dataset, detect_model_type


def evaluate_model(model_path='./models/cnn_best.h5', data_dir='./data', history=None):
    """
    Evaluate trained model and generate visualizations.
    Supports both CNN and transfer learning models.
    
    Args:
        model_path (str): Path to saved model file
                          Examples: './models/cnn_best.h5', './models/vgg16_best.h5'
        data_dir (str): Path to dataset directory
        history: Training history object (optional, for plotting training curves)
    """
    print("=" * 60)
    print("Model Evaluation - Multiclass Fish Image Classification")
    print("=" * 60)
    
    # Create reports directory
    os.makedirs('./reports', exist_ok=True)
    
    # Detect model type and determine target size
    model_name, target_size = detect_model_type(model_path)
    print(f"\nDetected model type: {model_name}")
    print(f"Using target size: {target_size}")
    
    # Load model
    print("\n[1/5] Loading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Reload dataset with appropriate target size
    print("\n[2/5] Loading validation dataset...")
    _, val_gen, num_classes = load_dataset(
        data_dir=data_dir,
        target_size=target_size
    )
    
    # Get class indices and labels
    class_indices = val_gen.class_indices
    class_labels = list(class_indices.keys())
    print(f"\nClass labels: {class_labels}")
    
    # Reset generator to ensure we get all validation data
    val_gen.reset()
    
    # Get predictions
    print("\n[3/5] Making predictions on validation set...")
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = val_gen.classes
    
    # Compute metrics
    print("\n[4/5] Computing evaluation metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score (macro and weighted averages)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )[0]
    recall_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )[1]
    f1_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )[2]
    precision_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[0]
    recall_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[1]
    f1_weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )[2]
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Display metrics in console
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    print(f"\nWeighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1-Score: {f1_weighted:.4f}")
    
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    for i, label in enumerate(class_labels):
        print(f"{label:30s} | Precision: {precision[i]:.4f} | "
              f"Recall: {recall[i]:.4f} | F1: {f1[i]:.4f} | "
              f"Support: {support[i]}")
    
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(class_report)
    
    # Generate and save visualizations
    print("\n[5/5] Generating visualizations...")
    
    # Generate model-specific report filenames
    model_prefix = model_name.lower().replace(' ', '_')
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name} - Fish Classification', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = f'./reports/confusion_matrix_{model_prefix}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {cm_path}")
    plt.close()
    
    # 2. Classification Report (Text File)
    report_path = f'./reports/classification_report_{model_prefix}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"CLASSIFICATION REPORT - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Macro Average:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1-Score: {f1_macro:.4f}\n\n")
        f.write("Weighted Average:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1-Score: {f1_weighted:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(class_report)
    print(f"Saved: {report_path}")
    
    # 3. Training Curves (if history is available)
    if history is not None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy curve
            axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
            axes[0].set_title(f'{model_name} - Model Accuracy', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Accuracy', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss curve
            axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
            axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
            axes[1].set_title(f'{model_name} - Model Loss', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            curves_path = f'./reports/training_curves_{model_prefix}.png'
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {curves_path}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot training curves: {e}")
    else:
        print("Training history not available. Skipping training curves plot.")
        print("(To plot training curves, pass history object to evaluate_model)")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    print(f"\nAll reports saved to: ./reports/")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': class_report
    }


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Model Evaluation - Select Model to Evaluate")
    print("=" * 60)
    
    # List available models
    models_dir = './models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if model_files:
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file}")
        else:
            print("\nNo model files found in ./models/")
            return None
    else:
        print("\nModels directory not found: ./models/")
        return None
    
    print("\nOptions:")
    print("  1. Enter model filename (e.g., cnn_best.h5 or vgg16_best.h5)")
    print("  2. Enter full path to model file")
    
    user_input = input("\nEnter model path or filename: ").strip()
    
    # Handle relative paths
    if not os.path.isabs(user_input):
        if not user_input.startswith('./models/'):
            model_path = os.path.join('./models', user_input)
        else:
            model_path = user_input
    else:
        model_path = user_input
    
    # Evaluate model
    results = evaluate_model(
        model_path=model_path,
        data_dir='./data',
        history=None  # Pass history object here if available
    )
    
    return results


if __name__ == "__main__":
    results = main()
