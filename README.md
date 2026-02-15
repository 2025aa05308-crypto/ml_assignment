<p align="center">
  <img src="https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/http://coursera-university-assets.s3.amazonaws.com/b9/c608c79b5c498a8fa55b117fc3282f/5.-Square-logo-for-landing-page---Alpha.png?auto=format%2Ccompress&dpr=1&w=180&h=180" width="120" alt="BITS Pilani">
</p>

<h2 align="center">BITS Pilani — Work Integrated Learning Programmes Division</h2>
<h3 align="center">M.Tech (AIML / DSE)</h3>
<h2 align="center">Machine Learning — Assignment 2</h2>

<p align="center">
  <b>Customer Churn Prediction using 6 Classification Models</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
</p>

<p align="center">
  <b>Student:</b> Akshaya Basayya Hiremath &nbsp;|&nbsp; <b>ID:</b> 2025AA05308
</p>

---
<p align="center">
  <b>Assignment URL</b> https://2025aa05308-mlassignment-2.streamlit.app/
</p>

---
## a. Problem Statement

Customer churn is a critical business challenge in the telecommunications industry where companies lose revenue when subscribers discontinue their services. This project builds and compares **six machine learning classification models** to predict whether a customer is likely to churn, enabling proactive retention strategies.

**Objectives:**
- Implement 6 different classification algorithms on the same dataset
- Evaluate each model using 6 standard metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Deploy an interactive Streamlit web application for real-time predictions
- Identify the best-performing model for churn prediction

---

## b. Dataset Description

| Property | Details |
|:---------|:--------|
| **Name** | Telco Customer Churn |
| **Source** | IBM Sample Datasets / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Instances** | 7,043 customers |
| **Features** | 19 input features + 1 target |
| **Target** | `Churn` (Binary: Yes / No) |
| **Class Split** | ~73.5% No Churn · ~26.5% Churn (imbalanced) |

### Feature Details

| # | Feature | Type | Description |
|:-:|---------|:----:|-------------|
| 1 | `customerID` | String | Unique identifier (dropped before training) |
| 2 | `gender` | Categorical | Male / Female |
| 3 | `SeniorCitizen` | Binary | 1 = Senior, 0 = Not |
| 4 | `Partner` | Categorical | Yes / No |
| 5 | `Dependents` | Categorical | Yes / No |
| 6 | `tenure` | Numeric | Months with the company |
| 7 | `PhoneService` | Categorical | Yes / No |
| 8 | `MultipleLines` | Categorical | Yes / No / No phone service |
| 9 | `InternetService` | Categorical | DSL / Fiber optic / No |
| 10 | `OnlineSecurity` | Categorical | Yes / No / No internet service |
| 11 | `OnlineBackup` | Categorical | Yes / No / No internet service |
| 12 | `DeviceProtection` | Categorical | Yes / No / No internet service |
| 13 | `TechSupport` | Categorical | Yes / No / No internet service |
| 14 | `StreamingTV` | Categorical | Yes / No / No internet service |
| 15 | `StreamingMovies` | Categorical | Yes / No / No internet service |
| 16 | `Contract` | Categorical | Month-to-month / One year / Two year |
| 17 | `PaperlessBilling` | Categorical | Yes / No |
| 18 | `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| 19 | `MonthlyCharges` | Numeric | Monthly charge amount ($) |
| 20 | `TotalCharges` | Numeric | Total charges to date ($) |
| **21** | **`Churn`** | **Binary** | **Target — Yes / No** |

---

## c. Models Used — Comparison Table

All six models were trained on the **same 80/20 train-test split** with consistent preprocessing.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------|:--------:|:---:|:---------:|:------:|:---:|:---:|
| **Logistic Regression** | 0.7991 | **0.8403** | 0.6426 | 0.5481 | 0.5916 | 0.4621 |
| **Decision Tree** | 0.7743 | 0.7647 | 0.5791 | 0.5481 | 0.5632 | 0.4115 |
| **K-Nearest Neighbor** | 0.7424 | 0.7603 | 0.5151 | 0.5027 | 0.5088 | 0.3343 |
| **Naive Bayes (Gaussian)** | 0.7466 | 0.8201 | 0.5160 | **0.7326** | **0.6055** | 0.4413 |
| **Random Forest (Ensemble)** | 0.7828 | 0.8275 | 0.6133 | 0.4920 | 0.5460 | 0.4098 |
| **XGBoost (Ensemble)** | 0.7928 | 0.8352 | **0.6367** | 0.5107 | 0.5668 | 0.4373 |

> **Best F1 Score:** Naive Bayes (0.6055) &nbsp;|&nbsp; **Best Accuracy & AUC:** Logistic Regression (79.91%, 0.8403) &nbsp;|&nbsp; **Best Recall:** Naive Bayes (73.26%)

---

## Observations on Model Performance

| ML Model Name | Observation |
|:--------------|:------------|
| **Logistic Regression** | Highest accuracy (79.91%) and best AUC (0.8403) among all models. Offers excellent balance of precision and recall with the second-best F1 score (0.5916). Strong overall performer for linearly separable patterns. Benefits from feature scaling. |
| **Decision Tree** | Moderate accuracy (77.43%) with the lowest AUC (0.7647). Reasonable precision (0.5791) but tends to overfit. F1 score of 0.5632 and MCC of 0.4115 indicate limited generalisation capability on unseen data. |
| **K-Nearest Neighbor** | Lowest accuracy (74.24%) and weakest F1 (0.5088) and MCC (0.3343). Distance-based approach struggles with the mixed feature types in this dataset despite feature scaling. Sensitive to the choice of k. |
| **Naive Bayes (Gaussian)** | **Best F1 score (0.6055)** and highest recall (73.26%) — catches the most churners. Slightly lower accuracy (74.66%) due to more false positives, but the strong MCC (0.4413) confirms reliable predictions. Ideal when minimising missed churners is the priority. |
| **Random Forest (Ensemble)** | Good accuracy (78.28%) and AUC (0.8275). Strong precision (0.6133) but conservative recall (49.20%). Ensemble bagging stabilises predictions but misses some churners. No scaling required. |
| **XGBoost (Ensemble)** | Second-best accuracy (79.28%) and AUC (0.8352). Balances precision (0.6367, highest) and recall (0.5107) well. A strong all-round gradient boosting approach with good generalisation. |

---

## Project Structure

```
project-folder/
├── app.py                       # Streamlit web application
├── ML_Assignment_2.ipynb        # Jupyter notebook with full analysis
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── sample_test_data.csv         # Sample CSV for testing the app
├── model_comparison.csv         # Model comparison results
├── model/                       # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── dataset/
    └── customer_churn.csv       # Telco Customer Churn dataset
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/2025aa05308-crypto/ml_assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook (optional — for training)
```bash
jupyter notebook ML_Assignment_2.ipynb
```
Execute all cells to train models, compute metrics, and save `.pkl` files.

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Streamlit App Features

| # | Required Feature | Status |
|:-:|:-----------------|:------:|
| 1 | Dataset upload option (CSV) — upload test data | ✅ |
| 2 | Model selection dropdown (6 models) | ✅ |
| 3 | Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) | ✅ |
| 4 | Confusion matrix & classification report | ✅ |

### Additional Features
- **BITS Pilani branding** — logo, student details, professional dark theme
- **Sample CSV download** — try the app instantly without preparing data
- **Model Comparison tab** — side-by-side metrics table, observations, bar charts
- **About tab** — problem statement, dataset info, key findings
- **Download predictions** — export results as CSV
- **Responsive design** — custom HTML/CSS styling

---

## Deployment on Streamlit Community Cloud

1. Push code to **GitHub** (ensure `requirements.txt`, `app.py`, `model/` are present)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → Sign in with GitHub
3. Click **New App** → select repository → branch `main` → file `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Results Summary

| Metric | Best Model | Value |
|:-------|:-----------|:-----:|
| Accuracy | Logistic Regression | **0.7991** |
| AUC | Logistic Regression | **0.8403** |
| Precision | XGBoost | **0.6367** |
| Recall | Naive Bayes (Gaussian) | **0.7326** |
| F1 Score | Naive Bayes (Gaussian) | **0.6055** |
| MCC | Logistic Regression | **0.4621** |

### Key Takeaway

**Naive Bayes (Gaussian)** achieves the best F1 score (0.6055) and highest recall (73.26%), making it the most effective model for identifying churners when balancing precision and recall. **Logistic Regression** leads in accuracy (79.91%) and AUC (0.8403), showing the strongest overall discriminative power. All models show moderate MCC values (0.33–0.46), reflecting the challenge of the ~73.5% / 26.5% class imbalance.

---

## Technologies Used

| Tool | Purpose |
|:-----|:--------|
| Python 3.8+ | Programming language |
| scikit-learn | Model training & evaluation |
| XGBoost | Gradient boosting classifier |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib & Seaborn | Visualisation |
| Streamlit | Web application framework |
| Joblib | Model serialisation |

---

## Student Details

**Akshaya Basayya Hiremath**  
- **ID:** 2025AA05308  
- **Programme:** M.Tech (AIML / DSE)  
- **Course:** Machine Learning  
- **Institution:** BITS Pilani — Work Integrated Learning Programmes Division

---

<p align="center"><i>Assignment completed on BITS Virtual Lab as per assignment guidelines.</i></p>
