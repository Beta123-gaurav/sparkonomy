# Engineering Adoption: Rogers' Diffusion Analytics Platform

### A Computational Framework for Sociometry and Data Engineering

**Version:** 1.0.0  
**Framework:** Streamlit, Scikit-Learn, XGBoost  
**Theory:** E.M. Rogers, *Diffusion of Innovations* (1962)

---

## 📖 Overview

This repository contains a production-grade analytics dashboard designed to operationalize Rogers' Diffusion of Innovations Theory. Historically, adoption research has been retrospective (analyzing why something *was* adopted). This platform shifts the paradigm to **predictive analytics**.

It allows stakeholders to ingest survey data regarding an innovation's attributes, train sophisticated ensemble machine learning models, and simulate future adoption scenarios using a "Prediction Playground."

## 🚀 Key Features

* **Smart Task Detection**: The system heuristically analyzes the target variable to automatically select the correct modeling strategy:
    * *Classification* (for Categories like "Innovator", "Laggard" or Likert Scales).
    * *Regression* (for Continuous metrics like "Time to Adoption").
* **Dual-Engine Learning**: Implements both **Random Forest** (Bagging) for variance reduction and **XGBoost** (Gradient Boosting) for bias reduction.
* **The Prediction Playground**: A dynamic simulation interface that generates UI widgets based on the training data schema, allowing users to test "What-If" scenarios (e.g., *"If we increase Trialability by 20%, does adoption probability increase?"*).
* **Robust Schema Alignment**: Solves the common "One-Hot Encoding" failure in production ML by enforcing training schema consistency during inference.

## 📂 Repository Structure

```text
├── app.py                  # The main Streamlit application logic
├── requirements.txt        # Python dependency list
├── requirements.md         # Detailed environment documentation
├── README.md               # Project documentation
└── data/                   # Folder for datasets (e.g., sparkonomy_synthetic_survey_30.xlsx)
