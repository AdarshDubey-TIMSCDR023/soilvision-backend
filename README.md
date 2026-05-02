# Soil Analysis & Crop Recommendation AI

A production-ready end-to-end machine learning pipeline that predicts the **best crop to grow** based on soil nutrients and environmental conditions — achieving **98.9% accuracy** across 22 crop classes.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [CLI Prediction](#cli-prediction)
- [Flask REST API](#flask-rest-api)
- [EDA Notebook](#eda-notebook)
- [Generated Plots](#generated-plots)
- [Tech Stack](#tech-stack)

---

## Overview

Given 7 soil and environmental parameters, the model recommends the optimal crop to cultivate.

| Input Feature | Description | Unit |
|---|---|---|
| `N` | Nitrogen content | kg/ha |
| `P` | Phosphorus content | kg/ha |
| `K` | Potassium content | kg/ha |
| `temperature` | Ambient temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH level | 0–14 |
| `rainfall` | Annual rainfall | mm |

**Output:** Recommended crop name + confidence score (e.g., `Rice — 92%`)

---

## Project Structure

```
soil_ai_model/
│
├── data/
│   └── crop_recommendation.csv     # 2,200 rows · 22 crops · 7 features
│
├── models/
│   ├── crop_model.pkl              # Trained RandomForestClassifier
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── label_encoder.pkl           # LabelEncoder (22 classes)
│
├── notebooks/
│   └── EDA.ipynb                   # Full exploratory data analysis
│
├── plots/
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── crop_distribution.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── train_model.py                  # 7-stage ML training pipeline
├── predict.py                      # CLI prediction script
├── app.py                          # Flask REST API
├── generate_dataset.py             # Synthetic dataset generator
└── requirements.txt
```

---

## Dataset

The pipeline uses the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle.

**To use the real dataset:**
1. Download `crop_recommendation.csv` from Kaggle
2. Place it in the `data/` folder

**To use the synthetic dataset** (included for demo):
```bash
python generate_dataset.py
```
This generates 2,200 samples across 22 crops using agronomically accurate parameter ranges.

---

## Installation

**Requirements:** Python 3.8+

```bash
# 1. Clone / download the project
cd soil_ai_model

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
flask>=3.0.0
shap>=0.44.0
```

---

## Quick Start

```bash
# Step 1 — Generate dataset (skip if using Kaggle CSV)
python generate_dataset.py

# Step 2 — Train the model
python train_model.py

# Step 3 — Predict a crop
python predict.py --N 90 --P 40 --K 40 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200

# Step 4 — Start the API
python app.py
```

---

## Pipeline Stages

`train_model.py` runs 7 sequential stages:

```
Stage 1 → Data Loading & Validation
          Load CSV, check schema, impute missing values

Stage 2 → Preprocessing
          StandardScaler + LabelEncoder + 80/20 stratified split

Stage 3 → Exploratory Data Analysis
          Save 4 EDA charts to plots/

Stage 4 → Multi-model Cross-validation
          5-fold CV on RandomForest, DecisionTree, SVM,
          GradientBoosting, KNN — compare Accuracy / F1

Stage 5 → Hyperparameter Tuning
          GridSearchCV on RandomForest:
          n_estimators × max_depth × min_samples_split × min_samples_leaf

Stage 6 → Final Evaluation
          Confusion matrix, classification report, feature importance,
          SHAP explainability (if shap installed)

Stage 7 → Save Artefacts
          crop_model.pkl · scaler.pkl · label_encoder.pkl
```

---

## Model Performance

Five models were trained and compared using 5-fold stratified cross-validation:

| Model | CV Accuracy | CV F1 |
|---|---|---|
| **RandomForest** | **98.98%** | **98.97%** |
| SVM | 97.67% | 97.63% |
| DecisionTree | 97.56% | 97.55% |
| GradientBoosting | 97.44% | 97.45% |
| KNN | 96.65% | 96.55% |

**Hold-out test set (20%):**

| Metric | Score |
|---|---|
| Accuracy | **98.86%** |
| Precision | **98.86%** |
| Recall | **98.86%** |
| F1 Score | **98.86%** |

**Feature Importance Ranking** (Random Forest Gini):

| Rank | Feature | Importance |
|---|---|---|
| 1 | rainfall | 0.368 |
| 2 | humidity | 0.201 |
| 3 | temperature | 0.148 |
| 4 | ph | 0.112 |
| 5 | K | 0.071 |
| 6 | N | 0.058 |
| 7 | P | 0.042 |

---

## CLI Prediction

```bash
python predict.py --N 90 --P 40 --K 40 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
```

**Output:**
```
╔══════════════════════════════════════════════════════╗
║       Crop Recommendation Predictor                  ║
╚══════════════════════════════════════════════════════╝

  ── Input Parameters ──────────────────────────────────
  N               : 90.0
  P               : 40.0
  K               : 40.0
  temperature     : 25.0
  humidity        : 80.0
  ph              : 6.5
  rainfall        : 200.0

  ── Prediction ────────────────────────────────────────
  Recommended Crop : RICE
  Confidence       : 66.00%

  ── Top-3 Candidates ──────────────────────────────────
  1. rice                66.00%  ███████████████████
  2. jute                17.00%  █████
  3. coffee              12.00%  ███
```

All parameters have default values, so `python predict.py` runs with the example above.

---

## Flask REST API

### Start the server

```bash
python app.py
# API available at http://localhost:5000
```

### Endpoints

#### `POST /predict`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 90,
    "phosphorus": 40,
    "potassium": 40,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
  }'
```

**Response:**
```json
{
  "crop": "rice",
  "confidence": 0.66,
  "top3": [
    {"crop": "rice",   "probability": 0.66},
    {"crop": "jute",   "probability": 0.17},
    {"crop": "coffee", "probability": 0.12}
  ]
}
```

#### `GET /health`

```bash
curl http://localhost:5000/health
```
```json
{"status": "ok", "uptime_s": 12.4, "timestamp": "2025-01-01T10:00:00Z"}
```

#### `GET /model-info`

```bash
curl http://localhost:5000/model-info
```
```json
{"model_type": "RandomForestClassifier", "n_features": 7, "n_estimators": 100}
```

### Input Validation

The API validates all inputs against agronomic bounds and returns `422` with a descriptive error on invalid values:

```json
{"error": "Validation error: 'ph' value 15.0 is out of range [0, 14]"}
```

### Production Deployment

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## EDA Notebook

Open `notebooks/EDA.ipynb` in Jupyter for interactive exploratory analysis:

```bash
pip install jupyter
jupyter notebook notebooks/EDA.ipynb
```

The notebook covers 17 sections including:
- Feature distributions with KDE overlays
- Outlier detection (IQR method)
- Pearson correlation heatmaps
- NPK nutrient profiles per crop
- Pair plots for feature interactions
- Radar charts for crop profiles
- pH zone analysis
- Rainfall vs temperature scatter
- NPK ternary plot
- Shapiro-Wilk normality tests
- Violin plots for class separability
- Key findings and modelling recommendations

---

## Generated Plots

All charts are saved to `plots/` after running `train_model.py`:

| File | Description |
|---|---|
| `feature_distributions.png` | Histogram + KDE for all 7 features |
| `correlation_heatmap.png` | Lower-triangle Pearson correlation matrix |
| `crop_distribution.png` | Sample count per crop label |
| `model_comparison.png` | 5-model CV accuracy & F1 bar chart |
| `feature_importance.png` | RF Gini importance, ranked |
| `confusion_matrix.png` | 22×22 confusion matrix on hold-out set |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `joblib` | Model serialisation |
| `matplotlib` | Chart rendering |
| `seaborn` | Statistical visualisations |
| `flask` | REST API server |
| `shap` | Model explainability (optional) |

---

## Supported Crops

rice · maize · chickpea · kidneybeans · pigeonpeas · mothbeans · mungbean · blackgram · lentil · pomegranate · banana · mango · grapes · watermelon · muskmelon · apple · orange · papaya · coconut · cotton · jute · coffee

---

*Built with scikit-learn · Flask · Matplotlib · Seaborn*
