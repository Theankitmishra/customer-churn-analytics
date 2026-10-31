"""
Customer Analytics ETL & Relational Data Pipeline
=================================================
Loads raw customer records, populates an indexed SQLite data warehouse,
executes analytical SQL queries for business intelligence, and exports
cleaned feature matrices for predictive modeling.

Author: Ankit Kumar Mishra
"""

import csv
import json
import os
import sqlite3
from typing import Dict, List, Tuple


def get_paths() -> Tuple[str, str, str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "customer_churn_50k.csv")
    db_path = os.path.join(base_dir, "data", "customer_warehouse.db")
    processed_csv = os.path.join(base_dir, "data", "processed_features.csv")
    metrics_json = os.path.join(base_dir, "data", "executive_kpis.json")
    return raw_csv, db_path, processed_csv, metrics_json


def initialize_warehouse(db_path: str):
    """Initializes the normalized relational schema with primary and foreign key constraints."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        gender TEXT,
        senior_citizen INTEGER,
        partner TEXT,
        dependents TEXT,
        tenure_months INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE contracts_services (
        customer_id TEXT PRIMARY KEY,
        phone_service TEXT,
        multiple_lines TEXT,
        internet_service TEXT,
        online_security TEXT,
        online_backup TEXT,
        device_protection TEXT,
        tech_support TEXT,
        streaming_tv TEXT,
        streaming_movies TEXT,
        contract_type TEXT,
        paperless_billing TEXT,
        payment_method TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE billing_telemetry (
        customer_id TEXT PRIMARY KEY,
        monthly_charges REAL,
        total_charges REAL,
        num_support_tickets INTEGER,
        avg_monthly_gb_usage REAL,
        churn TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
    );
    """)

    # Indices for fast analytical queries
    cursor.execute("CREATE INDEX idx_tenure ON customers(tenure_months);")
    cursor.execute("CREATE INDEX idx_contract ON contracts_services(contract_type);")
    cursor.execute("CREATE INDEX idx_churn ON billing_telemetry(churn);")

    conn.commit()
    conn.close()


def load_raw_data_to_sql(raw_csv: str, db_path: str):
    """Parses raw CSV rows and performs batch transactional insertion into SQL tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    customers_data = []
    contracts_data = []
    billing_data = []

    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["customer_id"]
            customers_data.append((
                cid,
                row["gender"],
                int(row["senior_citizen"]),
                row["partner"],
                row["dependents"],
                int(row["tenure_months"])
            ))

            contracts_data.append((
                cid,
                row["phone_service"],
                row["multiple_lines"],
                row["internet_service"],
                row["online_security"],
                row["online_backup"],
                row["device_protection"],
                row["tech_support"],
                row["streaming_tv"],
                row["streaming_movies"],
                row["contract_type"],
                row["paperless_billing"],
                row["payment_method"]
            ))

            billing_data.append((
                cid,
                float(row["monthly_charges"]),
                float(row["total_charges"]),
                int(row["num_support_tickets"]),
                float(row["avg_monthly_gb_usage"]),
                row["churn"]
            ))

    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers_data)
    cursor.executemany("INSERT INTO contracts_services VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", contracts_data)
    cursor.executemany("INSERT INTO billing_telemetry VALUES (?, ?, ?, ?, ?, ?)", billing_data)

    conn.commit()
    conn.close()
    print(f"Loaded {len(customers_data)} records into relational SQL warehouse: {db_path}")


def compute_executive_kpis(db_path: str, metrics_json: str) -> Dict:
    """Executes SQL analytical queries to compute core business health metrics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total customers & Churn overview
    cursor.execute("""
    SELECT 
        COUNT(*) as total_customers,
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
        AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100 as churn_rate,
        SUM(monthly_charges) as total_mrr,
        SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) as mrr_at_risk,
        AVG(monthly_charges) as avg_monthly_charge
    FROM billing_telemetry;
    """)
    totals = cursor.fetchone()

    # Churn rate grouped by contract type
    cursor.execute("""
    SELECT 
        c.contract_type,
        COUNT(*) as segment_count,
        ROUND(AVG(CASE WHEN b.churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate_pct,
        ROUND(AVG(b.monthly_charges), 2) as avg_charge
    FROM contracts_services c
    JOIN billing_telemetry b ON c.customer_id = b.customer_id
    GROUP BY c.contract_type
    ORDER BY churn_rate_pct DESC;
    """)
    contract_churn = [
        {"contract": r[0], "count": r[1], "churn_rate_pct": r[2], "avg_charge": r[3]}
        for r in cursor.fetchall()
    ]

    # Churn rate grouped by support tickets
    cursor.execute("""
    SELECT 
        b.num_support_tickets,
        COUNT(*) as count,
        ROUND(AVG(CASE WHEN b.churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate_pct
    FROM billing_telemetry b
    GROUP BY b.num_support_tickets
    ORDER BY b.num_support_tickets ASC;
    """)
    ticket_churn = [
        {"tickets": r[0], "count": r[1], "churn_rate_pct": r[2]}
        for r in cursor.fetchall()
    ]

    # Tenure Cohort Analysis
    cursor.execute("""
    SELECT 
        CASE 
            WHEN c.tenure_months <= 12 THEN '0-12 Months'
            WHEN c.tenure_months <= 24 THEN '13-24 Months'
            WHEN c.tenure_months <= 48 THEN '25-48 Months'
            ELSE '49+ Months'
        END AS tenure_cohort,
        COUNT(*) as cohort_size,
        ROUND(AVG(CASE WHEN b.churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate_pct,
        ROUND(SUM(CASE WHEN b.churn = 'Yes' THEN b.monthly_charges ELSE 0 END), 2) as cohort_mrr_loss
    FROM customers c
    JOIN billing_telemetry b ON c.customer_id = b.customer_id
    GROUP BY tenure_cohort
    ORDER BY churn_rate_pct DESC;
    """)
    cohort_churn = [
        {"cohort": r[0], "size": r[1], "churn_rate_pct": r[2], "mrr_loss": r[3]}
        for r in cursor.fetchall()
    ]

    conn.close()

    kpi_report = {
        "summary": {
            "total_customers": totals[0],
            "churned_customers": totals[1],
            "churn_rate_pct": round(totals[2], 2),
            "total_mrr": round(totals[3], 2),
            "mrr_at_risk": round(totals[4], 2),
            "avg_monthly_charge": round(totals[5], 2)
        },
        "by_contract": contract_churn,
        "by_support_tickets": ticket_churn,
        "by_tenure_cohort": cohort_churn
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(kpi_report, f, indent=2)

    print(f"Executive KPIs compiled and saved to: {metrics_json}")
    return kpi_report


def export_processed_features(raw_csv: str, output_csv: str):
    """One-hot encodes categoricals and normalizes numeric values for ML modeling."""
    processed_rows = []
    
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tenure = float(row["tenure_months"])
            monthly = float(row["monthly_charges"])
            total = float(row["total_charges"])
            tickets = float(row["num_support_tickets"])
            gb_usage = float(row["avg_monthly_gb_usage"])
            senior = 1.0 if row["senior_citizen"] == "1" else 0.0
            partner = 1.0 if row["partner"] == "Yes" else 0.0
            dependents = 1.0 if row["dependents"] == "Yes" else 0.0
            paperless = 1.0 if row["paperless_billing"] == "Yes" else 0.0

            # One-hot encodings
            is_m2m = 1.0 if row["contract_type"] == "Month-to-month" else 0.0
            is_one_yr = 1.0 if row["contract_type"] == "One year" else 0.0
            is_two_yr = 1.0 if row["contract_type"] == "Two year" else 0.0

            is_fiber = 1.0 if row["internet_service"] == "Fiber optic" else 0.0
            is_dsl = 1.0 if row["internet_service"] == "DSL" else 0.0
            is_no_net = 1.0 if row["internet_service"] == "No" else 0.0

            is_elec_check = 1.0 if row["payment_method"] == "Electronic check" else 0.0
            has_tech_support = 1.0 if row["tech_support"] == "Yes" else 0.0
            has_online_sec = 1.0 if row["online_security"] == "Yes" else 0.0

            churn_label = 1 if row["churn"] == "Yes" else 0

            processed_rows.append([
                row["customer_id"],
                tenure,
                monthly,
                total,
                tickets,
                gb_usage,
                senior,
                partner,
                dependents,
                paperless,
                is_m2m,
                is_one_yr,
                is_two_yr,
                is_fiber,
                is_dsl,
                is_no_net,
                is_elec_check,
                has_tech_support,
                has_online_sec,
                churn_label
            ])

    headers = [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "avg_monthly_gb_usage",
        "senior_citizen",
        "partner",
        "dependents",
        "paperless_billing",
        "contract_month_to_month",
        "contract_one_year",
        "contract_two_year",
        "internet_fiber_optic",
        "internet_dsl",
        "internet_none",
        "payment_electronic_check",
        "tech_support_yes",
        "online_security_yes",
        "churn"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(processed_rows)

    print(f"Exported ML feature matrix to: {output_csv}")


if __name__ == "__main__":
    raw_csv, db_path, processed_csv, metrics_json = get_paths()
    initialize_warehouse(db_path)
    load_raw_data_to_sql(raw_csv, db_path)
    compute_executive_kpis(db_path, metrics_json)
    export_processed_features(raw_csv, processed_csv)
