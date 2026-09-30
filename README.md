# 🛡️ Medical Insurance Cost Estimator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-insurance-cost-estimator.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64+-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?logo=scikit-learn)
![Model R²](https://img.shields.io/badge/Model%20R²-90.1%25-brightgreen)

> 🚀 **Live Web App**: Try the interactive predictor directly in your browser:  
> **👉 [https://medical-insurance-cost-estimator.streamlit.app/](https://medical-insurance-cost-estimator.streamlit.app/)**

---

An interactive actuarial machine learning web application that predicts personal annual medical insurance costs with **~90.1% accuracy ($R^2$)**, built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌟 Key Features

- **High-Accuracy ML Model**: Powered by a tuned `GradientBoostingRegressor` (150 estimators) yielding an $R^2 \approx 0.9007$ and Mean Absolute Error of ~\$2,040 on unseen data.
- **Graceful Healthcare UI**: Executive midnight-navy design with tailored jewel-toned risk cards (Low Risk Tier, Standard Tier, Elevated Risk Tier).
- **Comprehensive Demographic Inputs**:
  - Age (18–65)
  - Sex / Gender
  - BMI with **built-in interactive Height & Weight calculator** (supports Metric & Imperial)
  - Number of Children / Dependents (0–5+)
  - Smoking Status (with clear risk highlights)
  - US Region (Northeast, Northwest, Southeast, Southwest)
- **Instant Risk Classification**: Automatically assigns Low Risk 🟢, Standard Risk 🟡, or Elevated Risk 🔴 badges.
- **National Benchmark Comparison**: Compares estimates live against national and smoking cohort averages.
- **"What-If" Money-Saving Simulator**: Computes potential annual savings if the user quits smoking or reaches a healthy BMI range.
- **Interactive Lifetime Projection Curve**: Visualizes projected premiums from age 18 to 65 across different lifestyle risk tiers.

---

## 🌐 Live Access

You can use the application immediately without installing anything locally:

👉 **[Launch Medical Insurance Cost Estimator](https://medical-insurance-cost-estimator.streamlit.app/)**

---

## 📊 Model Performance Comparison

| Model | $R^2$ (Test Set) | MAE | Notes |
|---|---|---|---|
| Multiple Linear Regression Baseline | 71.8% | $3,756 | Simple linear model |
| Linear + Smart Interaction Features | 88.0% | $2,453 | Adds `obese_smoker` & `age_smoker` |
| **Gradient Boosting Regressor (Deployed)** | **90.1%** | **$2,040** | **Best performance; captures non-linear interactions automatically** |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sameer-04062004/Medical-Insurance-Cost-Estimator.git
cd Medical-Insurance-Cost-Estimator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

*(Windows users can also double-click `run_app.bat` to launch immediately).*

---

## 📁 Repository Structure

```
├── app.py                                         # Interactive Streamlit frontend
├── insurance_cost_model.joblib                    # Serialized trained model bundle
├── insurance.csv                                  # Training dataset (1,338 records)
├── sample_new_customers.csv                       # Batch evaluation samples
├── Medical_insurance_cost_prediction_clean.ipynb  # End-to-end Jupyter notebook
├── train_and_save_model.py                        # Standalone script to train & export model
├── run_app.bat                                    # 1-Click Windows launcher
├── launch_app.py                                  # Python launcher
├── .python-version                                # Pinned Python 3.11 environment
├── requirements.txt                               # Dependencies for deployment
└── README.md                                      # Documentation & Live Demo link
```

---

## ☁️ Deployment

The application is deployed on **Streamlit Community Cloud** with continuous integration enabled on the `main` branch:
- **Live URL**: [https://medical-insurance-cost-estimator.streamlit.app/](https://medical-insurance-cost-estimator.streamlit.app/)
- **Hosting Environment**: Python 3.11 on Linux
