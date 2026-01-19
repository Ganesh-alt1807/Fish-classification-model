# Fish Classification Model - Run Instructions

## Project Structure
This is a multi-component Fish Classification project with training, evaluation, and web deployment capabilities.

## Quick Start Guide

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Data**
Ensure your dataset is in the `./data` directory with the following structure:
```
data/
├── train/
│   ├── animal fish/
│   ├── animal fish bass/
│   └── ... (other classes)
├── val/
│   └── ... (same class structure)
└── test/
    └── ... (same class structure)
```

Alternatively, use the existing dataset in `./Dataset/data/`

### 3. **Train Models**

#### Option A: Train a CNN from scratch
```bash
python train.py
```
Then select option `1` when prompted.

#### Option B: Train Transfer Learning Models
```bash
python train.py
```
Then select option `2` and choose models (e.g., VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).

### 4. **Evaluate Models**
```bash
python evaluate.py
```
This will generate metrics, confusion matrices, and visualizations in the `./reports` directory.

### 5. **Run the Web Application**
**Important:** The app.py file is a Streamlit application and MUST be run with streamlit, not python.

```bash
streamlit run app.py
```

This will:
- Open a web browser with the Fish Classification interface
- Allow you to upload images and get predictions
- Show confidence scores and top predictions

## Project Files

- **train.py** - Train models (CNN or transfer learning)
- **app.py** - Streamlit web application (USE: `streamlit run app.py`)
- **evaluate.py** - Model evaluation and metrics generation
- **utils.py** - Utility functions for data loading and model building
- **requirements.txt** - Python dependencies

## Troubleshooting

### Issue: "Models directory not found"
- Make sure to train a model first using `python train.py`
- Check that models are saved in `./models/` directory

### Issue: "No class labels found"
- Ensure the `./data` directory exists with class subdirectories
- Or update `load_dataset()` in utils.py to point to your dataset location

### Issue: Streamlit errors when running app.py
- Use `streamlit run app.py` NOT `python app.py`
- Streamlit requires the `streamlit run` command to function properly

## Supported Transfer Learning Models
- VGG16
- ResNet50
- MobileNet
- InceptionV3
- EfficientNetB0
