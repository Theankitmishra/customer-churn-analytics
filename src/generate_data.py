"""
Synthetic Customer Churn Dataset Generator (50,000 Records)
============================================================
Generates realistic customer demographic, contract, and behavioral data
with realistic correlations for churn modeling and business analytics.

Author: Ankit Kumar Mishra
"""

import csv
import math
import os
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def generate_dataset(output_path: str, num_records: int = 50000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = [
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure_months",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract_type",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "avg_monthly_gb_usage",
        "churn"
    ]

    contracts = ["Month-to-month", "One year", "Two year"]
    contract_weights = [0.55, 0.25, 0.20]

    internet_types = ["Fiber optic", "DSL", "No"]
    internet_weights = [0.45, 0.35, 0.20]

    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    payment_weights = [0.35, 0.20, 0.25, 0.20]

    rows = []
    
    for i in range(1, num_records + 1):
        cid = f"CUST-{i:06d}"
        gender = random.choice(["Male", "Female"])
        senior = 1 if random.random() < 0.16 else 0
        partner = "Yes" if random.random() < 0.48 else "No"
        dependents = "Yes" if (partner == "Yes" and random.random() < 0.60) else "No"

        # Tenure: skewed towards early months + mature clusters
        if random.random() < 0.35:
            tenure = random.randint(1, 12)
        elif random.random() < 0.65:
            tenure = random.randint(13, 48)
        else:
            tenure = random.randint(49, 72)

        contract = random.choices(contracts, weights=contract_weights)[0]
        internet = random.choices(internet_types, weights=internet_weights)[0]
        payment = random.choices(payments, weights=payment_weights)[0]
        paperless = "Yes" if random.random() < 0.59 else "No"

        has_phone = "Yes" if random.random() < 0.90 else "No"
        multiple_lines = "No"
        if has_phone == "Yes":
            multiple_lines = "Yes" if random.random() < 0.42 else "No"
        else:
            multiple_lines = "No phone service"

        if internet != "No":
            online_sec = "Yes" if random.random() < 0.35 else "No"
            online_bkp = "Yes" if random.random() < 0.40 else "No"
            dev_prot = "Yes" if random.random() < 0.38 else "No"
            tech_sup = "Yes" if random.random() < 0.33 else "No"
            stream_tv = "Yes" if random.random() < 0.44 else "No"
            stream_mov = "Yes" if random.random() < 0.45 else "No"
            gb_usage = round(random.gauss(240, 60), 1) if internet == "Fiber optic" else round(random.gauss(110, 35), 1)
            gb_usage = max(10.0, min(800.0, gb_usage))
        else:
            online_sec = online_bkp = dev_prot = tech_sup = stream_tv = stream_mov = "No internet service"
            gb_usage = 0.0

        # Base monthly charge calculations
        base_charge = 20.0
        if has_phone == "Yes":
            base_charge += 10.0
            if multiple_lines == "Yes":
                base_charge += 15.0
        if internet == "DSL":
            base_charge += 30.0
        elif internet == "Fiber optic":
            base_charge += 50.0

        addon_services = [online_sec, online_bkp, dev_prot, tech_sup, stream_tv, stream_mov]
        addons_count = sum(1 for s in addon_services if s == "Yes")
        base_charge += (addons_count * 5.0)
        
        # Add slight natural pricing noise
        monthly_charge = round(base_charge + random.uniform(-3.0, 3.0), 2)
        monthly_charge = max(18.25, min(120.0, monthly_charge))

        # Support tickets
        if contract == "Month-to-month":
            support_tickets = random.choices([0, 1, 2, 3, 4, 5], weights=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04])[0]
        else:
            support_tickets = random.choices([0, 1, 2, 3], weights=[0.70, 0.20, 0.07, 0.03])[0]

        # Total charges (tenure * monthly charge with slight billing variance)
        total_charges = round(tenure * monthly_charge * random.uniform(0.97, 1.03), 2)

        # Churn Probability Score (Log-odds calculation)
        log_odds = -1.2
        if contract == "Month-to-month":
            log_odds += 1.35
        elif contract == "Two year":
            log_odds -= 1.40

        if tenure <= 6:
            log_odds += 1.10
        elif tenure >= 48:
            log_odds -= 1.20

        if payment == "Electronic check":
            log_odds += 0.45
        if internet == "Fiber optic":
            log_odds += 0.35
        if tech_sup == "No" and internet != "No":
            log_odds += 0.50
        if support_tickets >= 3:
            log_odds += (support_tickets * 0.45)
        if monthly_charge > 85:
            log_odds += 0.30

        prob = 1.0 / (1.0 + math.exp(-log_odds))
        churn = "Yes" if random.random() < prob else "No"

        rows.append([
            cid, gender, senior, partner, dependents, tenure,
            has_phone, multiple_lines, internet, online_sec,
            online_bkp, dev_prot, tech_sup, stream_tv, stream_mov,
            contract, paperless, payment, monthly_charge,
            total_charges, support_tickets, gb_usage, churn
        ])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {num_records} records successfully at: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "..", "data", "customer_churn_50k.csv")
    generate_dataset(target_csv, 50000)
