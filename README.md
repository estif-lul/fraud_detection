# 🔍 Fraud Detection in E-Commerce & Banking Transactions

## 📌 Project Overview

This repository contains a machine learning project aimed at detecting fraudulent transactions in both **e-commerce** and **banking** environments. As a data scientist at **Adey Innovations Inc.**, the task is to engineer features, handle imbalanced data, build interpretable models, and evaluate performance using business-driven metrics. The focus is on accuracy, explainability, and practicality in real-world fraud detection systems.

---

## 🗃️ Dataset Summary

### 🛒 `Fraud_Data.csv` (E-Commerce)
- Includes timestamps, purchase details, device/browser metadata, IP address, and user demographics.
- Target: `class` → 1 = Fraudulent, 0 = Legitimate
- Challenge: Highly imbalanced (~3% fraud)

### 💳 `creditcard.csv` (Banking)
- 28 anonymized features (`V1`–`V28`) from PCA + `Amount`, `Time`, and fraud label
- Target: `Class` → 1 = Fraudulent, 0 = Legitimate
- Challenge: Extremely imbalanced (~0.17% fraud)

### 🌍 `IpAddress_to_Country.csv`
- Maps IP ranges to country for geolocation-based feature engineering.

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning
- Deduplication, timestamp correction, mixed type resolution
- Anomaly flags (e.g. rapid-repeat transactions, high-frequency devices)
- IP-to-country mapping using integer transformation
- Outlier detection via IQR for transaction amount

### 2️⃣ Feature Engineering
- E-commerce: `time_since_signup`, `hour_of_day`, `day_of_week`, `device txn_count`
- Flags for suspicious patterns (`is_rapid_repeat`, `is_high_freq_device`, etc.)
- Banking: Focus on scaling `Amount` + temporal insights from `Time`

### 3️⃣ Handling Class Imbalance
- Applied **SMOTE** oversampling on training set only
- Justified by preserving feature distribution and minority signal amplification

### 4️⃣ Scaling & Encoding
- Scaled numerical features (`Amount`) using `StandardScaler`
- Categorical features one-hot encoded for modeling

### 5️⃣ Modeling
- **Baseline:** Logistic Regression
- **Advanced:** XGBoost (chosen for ensemble power and handling imbalance)
- Evaluated using:
  - **AUC-PR**
  - **F1-Score**
  - Confusion Matrix (for false positive trade-off)

### 6️⃣ Model Explainability
- Used **SHAP** to interpret XGBoost decisions
- Summary and force plots reveal global and local feature impacts
- Key insights include transaction time behavior, purchase value anomalies, and device/browser patterns

---

## 📁 Repository Structure

```bash
fraud_detection/
├── data/                  # Raw and processed datasets
├── notebooks/             # EDA, preprocessing, modeling, SHAP analysis
├── src/                   # Modular scripts (cleaning, modeling, evaluation)
├── models/                # Saved models and results
├── reports/               # Interim reports, visuals, final blog/pdf
├── requirements.txt       # Python dependencies
└── README.md              # This file

```
---

## 📦 Installation Guide

```bash
# Clone the repo
git clone https://github.com/your-username/fraud_detection.git
cd fraud_detection

# Create virtual environment
python -m venv .venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
---

# Install dependencies
pip install -r requirements.txt

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

---

## 📧 Contact

For questions or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

## 📜 License

This project is licensed under the Apache License. See [LICENSE](LICENSE) for details.

---