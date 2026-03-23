# Santander Customer Transaction Prediction

GWU DATS 6202 - Machine Learning I (Spring 2026) | Homework 2

## Overview

Binary classification pipeline to predict whether a Santander bank customer will make a specific transaction, based on 200 anonymized numerical features.

- **Competition**: [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction)
- **Task**: Binary classification (0 = no transaction, 1 = transaction)
- **Evaluation metric**: F1-macro
- **Baseline**: 0.670522 → Achieved: **0.690627** ✅

## Pipeline

1. **Data Preprocessing** — train/val/test split, identifier removal, missing value handling, MinMaxScaler
2. **Hyperparameter Tuning** — MLPClassifier with GridSearchCV + PredefinedSplit
3. **Model Selection** — best model selected by F1-macro score

## Model

- **MLPClassifier** (Shallow Neural Network)
- Best params: `hidden_layer_sizes=(200, 100), alpha=0.0001, learning_rate_init=0.01`

## Files

| File | Description |
|------|-------------|
| `Homework_2.ipynb` | Main homework notebook |
| `case_study.ipynb` | Reference pipeline (provided by instructor) |
| `pmlm_utilities_shallow.ipynb` | Utility functions |
| `pmlm_models_shallow.ipynb` | Model implementations |

## Requirements

Run on Google Colab with Google Drive mounted.
