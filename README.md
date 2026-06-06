# 💳 Credit Card Fraud Detection & Explainable AI Pipeline

## 📌 Project Overview
This project is an end-to-end machine learning pipeline designed to detect fraudulent credit card transactions. Instead of relying on a black-box model, this system emphasizes **data imbalance handling** and **mathematical explainability**, culminating in an interactive web dashboard for real-time risk analysis.

## 🚧 The Challenge: Extreme Class Imbalance
The dataset contains European credit card transactions over a two-day period. Out of 284,807 transactions, only 492 are fraudulent. 
* **Fraud Rate:** 0.172%
* **The Problem:** A standard model would achieve 99.8% accuracy by simply guessing "Normal" every time, catching zero actual fraud.

## 🛠️ The Solution & Architecture
To build a production-ready risk tool, this pipeline implements four core stages:

1. **Imbalance Mitigation (SMOTE):** Applied Synthetic Minority Over-sampling Technique strictly to the training set to generate intelligent, synthetic fraud cases, bringing the training class distribution to a perfect 50/50 split without leaking data into the test set.
2. **Predictive Modeling (Random Forest):** Trained a Random Forest Classifier on the balanced data. The model successfully navigates the complex, non-linear vectors (PCA features V1-V28) to identify hidden fraud patterns.
   * *Performance:* Achieved **83% Recall** and **84% Precision** on the untouched, highly imbalanced test set.
3. **Model Explainability (SHAP):** Integrated SHapley Additive exPlanations to eliminate the "black-box" effect. The explainer reverse-engineers the model's predictions to show exactly which behavioral vectors drove a specific fraud score.
4. **Interactive Deployment (Streamlit):** Wrapped the trained model and SHAP explainer into a lightweight web application, allowing non-technical stakeholders to simulate transactions and view dynamic risk assessments.

## 💻 Tech Stack
* **Languages:** Python (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Explainable AI:** SHAP
* **Visualization & UI:** Matplotlib, Seaborn, Streamlit

## 🚀 How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `python -m streamlit run app.py`