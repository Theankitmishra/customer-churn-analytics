# 📊 Customer Churn & Business Performance Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/SQL-PostgreSQL%20%7C%20SQLite-003B57.svg)](https://sqlite.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-v1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-5%20Passing-brightgreen.svg)](tests/)

An enterprise-scale data analytics, warehouse ETL, and predictive churn modeling platform engineered in **Python** and **SQL**. Analyzes **50,000+ customer records** to identify revenue drop-off patterns, track tenure cohorts, and predict customer churn risk using machine learning.

Developed by **Ankit Kumar Mishra** (M.Tech Computer Science, Delhi Technological University).

---

## 📌 Executive Summary & Key Findings

| Metric | Measured Value | Business Interpretation |
| :--- | :--- | :--- |
| **Total Active Customers Analyzed** | **50,000** | Full portfolio analyzed across 18 demographic & transactional attributes |
| **Total Monthly Recurring Revenue (MRR)** | **\$3,842,280** | Baseline portfolio revenue run-rate |
| **MRR at Risk (Predicted Churn)** | **\$2,080,939** | High-priority revenue at risk requiring proactive customer success outreach |
| **Primary Churn Model (Random Forest)** | **0.8228 ROC-AUC** | 74.3% accuracy, 78.4% recall on unseen test holdout (10,000 customers) |
| **Baseline Model (Logistic Regression)** | **0.8138 ROC-AUC** | 73.4% accuracy, 77.3% recall |

### 🔍 Top 3 Analytical Insights:
1. **Contract Commitment is the #1 Retention Lever:** Month-to-month contract customers exhibit a **71.32% churn rate**, compared to **16.10%** for customers on 2-year agreements (>4.4x difference).
2. **Support Ticket Escalation Cliff:** Customers with 0 support tickets churn at **44.1%**, while customers submitting **3 or more tickets surge to >81.0% churn probability**.
3. **Early Tenure Vulnerability:** The **0–12 month tenure cohort** accounts for **\$881,952 (42.4%)** of all lost MRR.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────┐
│  50,000 Raw Customer Records    │
│  (Demographics, Billing, Usage) │
└────────────────┬────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  SQL Relational Warehouse (data/customer_warehouse.db) │
│  • customers (customer_id PK, tenure, demographics)   │
│  • contracts_services (contract_type, addons, payment) │
│  • billing_telemetry (monthly_charges, tickets, churn) │
└────────────────┬───────────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐   ┌────────────────────────────────────────┐
│ SQL Cohorts  │   │ ML Feature Extraction & Encoding       │
│ & KPI Engine │   │ (StandardScaler, One-Hot Categoricals) │
└──────┬───────┘   └───────────────────┬────────────────────┘
       │                               │
       │                               ▼
       │                   ┌────────────────────────┐
       │                   │ Model Training Engine  │
       │                   │ • Random Forest (0.82) │
       │                   │ • Logistic Regression  │
       │                   └───────────┬────────────┘
       │                               │
       ▼                               ▼
┌────────────────────────────────────────────────────────┐
│           Interactive Streamlit Dashboard              │
│  • Executive KPI Scorecard                             │
│  • Tenure & Support Ticket Cohort Analysis             │
│  • Real-Time Churn Risk & Retention Simulator          │
└────────────────────────────────────────────────────────┘
```

---

## 🤖 Predictive Model Evaluation (Test Holdout: 10,000 Customers)

| Metric | Logistic Regression | Random Forest Classifier | Delta |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | `0.8138` | **`0.8228`** | **+0.0090** |
| **Accuracy** | `73.42%` | **`74.25%`** | **+0.83%** |
| **Precision** | `0.7334` | **`0.7396`** | **+0.0062** |
| **Recall (Sensitivity)** | `0.7727` | **`0.7836`** | **+0.0109** |
| **F1-Score** | `0.7525` | **`0.7609`** | **+0.0084** |

### Top Churn Drivers (Random Forest Feature Importances):
1. **`contract_month_to_month` (21.33%)** — High short-term contract flexibility increases churn.
2. **`contract_two_year` (18.23%)** — Strongest protective factor against customer cancellation.
3. **`tenure_months` (14.04%)** — Customers past the 24-month mark demonstrate strong stickiness.
4. **`total_charges` (10.37%)** — Lifetime spend correlation.
5. **`num_support_tickets` (9.54%)** — Service dissatisfaction and technical friction indicator.
6. **`monthly_charges` (7.48%)** — Sensitivity to price points above \$85/month.

---

## 📁 Repository Structure

```
customer-churn-analytics/
├── data/
│   ├── customer_churn_50k.csv         # Raw 50,000 customer dataset
│   ├── customer_warehouse.db          # Normalized SQLite relational warehouse
│   ├── processed_features.csv         # Cleaned & encoded ML feature matrix
│   └── executive_kpis.json            # Precomputed SQL cohort & KPI aggregations
├── models/
│   ├── model_evaluation_metrics.json  # Exported test metrics, confusion matrices, weights
│   ├── rf_model.pkl                   # Trained Random Forest classifier
│   └── scaler.pkl                     # Fitted StandardScaler instance
├── src/
│   ├── generate_data.py               # Generates synthetic 50,000-record dataset
│   ├── etl.py                         # SQL database ingestion, KPI calculation & ETL
│   ├── train.py                       # ML model training, cross-validation & evaluation
│   └── app.py                         # Interactive Streamlit analytics dashboard
├── tests/
│   └── test_data_pipeline.py          # Unit test suite verifying schemas and metrics
├── Dockerfile                         # Container definition for zero-config deployment
├── requirements.txt                   # Production dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Theankitmishra/customer-churn-analytics.git
cd customer-churn-analytics

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
# Generate 50,000 records
python3 src/generate_data.py

# Ingest into SQL warehouse and run analytical queries
python3 src/etl.py

# Train ML models and export performance metrics
python3 src/train.py
```

### 3. Launch Interactive Dashboard
```bash
streamlit run src/app.py
```
Open `http://localhost:8501` to view the live dashboard and Churn Risk Simulator.

### 4. Run Automated Test Suite
```bash
python3 -m unittest discover tests/
```

---

## 🐳 Docker Deployment

```bash
docker build -t customer-churn-analytics .
docker run -p 8501:8501 customer-churn-analytics
```

---

## 👨‍💻 Author

**Ankit Kumar Mishra**  
- M.Tech in Computer Science, **Delhi Technological University (DTU)**
- Portfolio: [portfolio-virid-nu-3wnb5oe7nu.vercel.app](https://portfolio-virid-nu-3wnb5oe7nu.vercel.app)
- GitHub: [@Theankitmishra](https://github.com/Theankitmishra)
- LinkedIn: [linkedin.com/in/theankitmishraa](https://www.linkedin.com/in/theankitmishraa/)
