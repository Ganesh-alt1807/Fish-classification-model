# Multiclass Fish Image Classification

## Overview
This project classifies fish images into multiple species using deep learning.
It uses two approaches:
- A custom CNN built from scratch
- Transfer learning using VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0

The final deployment is done via Streamlit, allowing users to upload an image and receive real-time predictions.

---

## Dataset
- The dataset contains images organized into subfolders by fish species.
- Place your dataset inside the `./data/` directory.
- The dataset is not included in the repository due to size constraints.

Example:
data/
├─ Species_A/
├─ Species_B/
├─ Species_C/
...


---

## Project Structure

project/
├─ data/ # dataset (not in repo)
├─ models/ # saved .h5 models
├─ notebooks/ # research & experiments
├─ reports/ # metrics, confusion matrix, results
├─ src/ # helper code
│ ├─ utils.py
├─ train.py # training scripts (cnn + transfer)
├─ evaluate.py # evaluation scripts
├─ app.py # streamlit deployment
├─ requirements.txt
├─ .gitignore
└─ README.md


---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy / Pandas
- scikit-learn
- Matplotlib / Seaborn
- Streamlit

---

## Train Models

### Train CNN:

python train.py --model cnn


### Train Transfer Models:
Inside `train.py`, specify:

models = ["VGG16", "ResNet50"]


Then run:

python train.py --transfer


---

## Evaluation

Evaluate any saved model:

python evaluate.py --model models/cnn_best.h5


Metrics reported:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Outputs saved to:
./reports/


---

## Deployment (Streamlit)
Run the Streamlit app:

streamlit run app.py


Features:
- Upload image
- Choose model
- View predicted species + confidence

---

## Results & Comparison
Transfer learning results stored in:
reports/transfer_results.json


---

## GitHub Guidelines
- Do not upload `data/` (too large)
- `models/` may contain placeholder only
- Keep repository clean and modular

---

## Future Improvements
- ONNX model export
- FastAPI backend deployment
- Additional fish species
- Hyperparameter tuning

