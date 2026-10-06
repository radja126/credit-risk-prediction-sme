# 💳 An AI-Based Credit Risk Prediction Model for SME Financing

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

---

## 📌 Executive Summary
In the financial industry, inaccurate credit risk assessment can lead to severe financial losses due to loan defaults. This project develops an end-to-end Machine Learning pipeline to predict loan default risks using the **Lending Club Loan Dataset** (10,000 records). 

By implementing **SMOTE** to handle severe class imbalance and performing extensive **Feature Engineering**, three classification models (**Decision Tree**, **Logistic Regression**, and **XGBoost**) were trained and evaluated. Using **Explainable AI (XAI)**, key financial and credit historical features driving default risks were identified to support proactive risk management decisions.

---

## 🎯 Business Problem & Key Objectives
- **Problem:** Financial institutions face high default risks (*Charged Off*) which severely impact portfolio profitability. Identifying potential defaulters early is more critical than overall prediction accuracy.
- **Goal:** Build a robust binary classification model that maximizes **Recall** (detecting actual default cases) while maintaining model interpretability for credit scoring compliance.

---

## 🛠️ Data Preprocessing & Pipeline
1. **Data Cleaning & Imputation:** Handled missing values across numerical and categorical features using mode imputation (`mort_acc`, `emp_title`, `emp_length`, etc.).
2. **Feature Engineering:** Derived 9 business-driven synthetic features, including:
   - `credit_history_age_months`: Borrower credit history length.
   - `loan_to_income`: Ratio of loan amount against annual income.
   - `credit_to_income`: Ratio of credit history age against annual income.
3. **Encoding:** Applied One-Hot Encoding to categorical variables (resulting in 106 clean numerical features).
4. **Class Imbalance Handling:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set to balance the minority target class (`Charged Off`).

---

## 📊 Model Evaluation & Comparison

Models were evaluated using an 80:20 Train-Test Hold-Out split and validated via **5-Fold Stratified Cross-Validation**.

| Model | Test Accuracy | Test Precision | **Test Recall** | **Test F1-Score** | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 77.55% | 38.83% | 29.42% | 33.48% | Stable |
| **Logistic Regression** 🌟 | **65.30%** | **31.05%** | **66.14%** | **42.26%** | **Stable (Selected)** |
| **XGBoost** | 80.15% | 42.69% | 9.89% | 16.06% | Overfitting Risk |

> **Key Finding:** While XGBoost achieved the highest overall accuracy (80.15%), **Logistic Regression** was selected as the optimal model for credit scoring because it achieved the highest **Recall (66.14%)**, ensuring the maximum detection of potential default cases.

---

## 🔍 Explainable AI (XAI) Insights
Feature importance and coefficient analysis revealed the top drivers influencing loan default decisions:
- **Loan Grade (`grade_C`, `grade_A`, `grade_D`):** Primary indicator of borrower creditworthiness.
- **Interest Rate (`int_rate`):** Higher interest rates strongly correlate with higher default probability.
- **Home Ownership & Income Ratios (`credit_to_income`):** Validated that credit age relative to annual income provides significant predictive power for risk assessment.

---

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (`imblearn`), XGBoost, Matplotlib, Seaborn

---

## 👥 Authors (Kelompok FARHAN)
- **Radja Triyasa** - *BINUS University*
- **Farhan Yahya Hadil**
- **M. Fadzli Nur Arrafi**
- **Muhammad Danielo Hadiyanto**

*Supervised by: D7053 - Anindhita Dewabharata S.Kom., M.Kom., Ph.D.*