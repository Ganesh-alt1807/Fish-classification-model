#!/usr/bin/env python
"""
Quick Start Script for Fish Classification Model
This script helps you get started with the project immediately.
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def print_info(text):
    print(f"\n>>> {text}")

def check_data_directory():
    """Check if dataset is available"""
    print_info("Checking for dataset...")
    
    if os.path.exists('./Dataset/data/train'):
        print("Found dataset at ./Dataset/data")
        return True
    elif os.path.exists('./data/train'):
        print("Found dataset at ./data")
        return True
    else:
        print("ERROR: Dataset not found!")
        print("Please ensure the dataset is in either:")
        print("  - ./Dataset/data/train, ./Dataset/data/val, ./Dataset/data/test")
        print("  - ./data/ with class subfolders")
        return False

def check_models_directory():
    """Check if models directory exists"""
    os.makedirs('./models', exist_ok=True)
    print_info("Models directory ready at ./models")

def main():
    print_header("Fish Classification Model - Quick Start")
    
    print("\n1. DEPENDENCY CHECK")
    print("-" * 40)
    print("Verifying all dependencies are installed...")
    
    try:
        import tensorflow
        print("OK: TensorFlow installed")
    except ImportError:
        print("ERROR: TensorFlow not installed. Run: pip install -r requirements.txt")
        return
    
    try:
        import streamlit
        print("OK: Streamlit installed")
    except ImportError:
        print("ERROR: Streamlit not installed. Run: pip install -r requirements.txt")
        return
    
    print("\n2. DATA CHECK")
    print("-" * 40)
    if not check_data_directory():
        return
    
    print("\n3. DIRECTORY SETUP")
    print("-" * 40)
    check_models_directory()
    
    print_header("Quick Start Options")
    
    print("""
Choose an option:
    
    INTERACTIVE MODE:
    1. Train a CNN model from scratch
       Command: python train.py
       (Will prompt for input)
       
    2. Train transfer learning models
       Command: python train.py
       (Will prompt for model selection)
       
    COMMAND-LINE MODE (Non-Interactive):
    3. Train CNN model (non-interactive)
       Command: python train.py --cnn --epochs 20
       
    4. Train specific transfer models
       Command: python train.py --transfer VGG16 ResNet50 --epochs 10
       
    5. Train all transfer models
       Command: python train.py --transfer all --epochs 20
       
    6. Evaluate an existing model
       Command: python evaluate.py
       
    7. Run the web app for predictions
       Command: streamlit run app.py
       
    8. View help and all options
       Command: python train.py --help

Tips:
  - Start with option 1 or 3 for quick testing (CNN is faster)
  - Use option 5 for best accuracy (all transfer learning models)
  - Training time depends on epochs and your GPU/CPU
  - Once you have a model, use option 7 for the web interface
  - The web app will automatically detect available models
    """)

if __name__ == "__main__":
    main()
