"""
Unit & Integration Tests for Customer Analytics Pipeline
========================================================
Validates dataset integrity, database constraints, ETL feature transformations,
and model metric thresholds.

Author: Ankit Kumar Mishra
"""

import json
import os
import sqlite3
import unittest


class TestCustomerAnalyticsPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.raw_csv = os.path.join(cls.base_dir, "data", "customer_churn_50k.csv")
        cls.db_path = os.path.join(cls.base_dir, "data", "customer_warehouse.db")
        cls.processed_csv = os.path.join(cls.base_dir, "data", "processed_features.csv")
        cls.kpi_json = os.path.join(cls.base_dir, "data", "executive_kpis.json")
        cls.metrics_json = os.path.join(cls.base_dir, "models", "model_evaluation_metrics.json")

    def test_01_raw_dataset_exists_and_has_50k_rows(self):
        self.assertTrue(os.path.exists(self.raw_csv), "Raw 50k CSV dataset must exist")
        with open(self.raw_csv, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        self.assertEqual(lines, 50001, "Expected 50,000 rows plus 1 header line")

    def test_02_sql_warehouse_schema_and_counts(self):
        self.assertTrue(os.path.exists(self.db_path), "SQLite warehouse must exist")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM customers;")
        cust_count = cur.fetchone()[0]
        self.assertEqual(cust_count, 50000)

        cur.execute("SELECT COUNT(*) FROM contracts_services;")
        contract_count = cur.fetchone()[0]
        self.assertEqual(contract_count, 50000)

        cur.execute("SELECT COUNT(*) FROM billing_telemetry;")
        billing_count = cur.fetchone()[0]
        self.assertEqual(billing_count, 50000)

        # Integrity test: Check for null IDs
        cur.execute("SELECT COUNT(*) FROM customers WHERE customer_id IS NULL;")
        null_ids = cur.fetchone()[0]
        self.assertEqual(null_ids, 0)

        conn.close()

    def test_03_processed_feature_matrix_validity(self):
        self.assertTrue(os.path.exists(self.processed_csv), "Processed features must exist")
        import csv
        with open(self.processed_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            
            # Check expected columns
            expected = ["tenure_months", "monthly_charges", "num_support_tickets", "churn"]
            for col in expected:
                self.assertIn(col, first_row)

            # Check churn label is binary (0 or 1)
            self.assertIn(first_row["churn"], ("0", "1"))

    def test_04_executive_kpis_validity(self):
        self.assertTrue(os.path.exists(self.kpi_json), "KPI JSON must exist")
        with open(self.kpi_json, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        self.assertEqual(summary.get("total_customers"), 50000)
        self.assertGreater(summary.get("churn_rate_pct"), 0.0)
        self.assertLess(summary.get("churn_rate_pct"), 100.0)
        self.assertGreater(summary.get("total_mrr"), 1000000.0)

    def test_05_predictive_model_metrics(self):
        self.assertTrue(os.path.exists(self.metrics_json), "Model metrics JSON must exist")
        with open(self.metrics_json, "r") as f:
            data = json.load(f)

        rf = data.get("random_forest", {})
        self.assertGreater(rf.get("roc_auc", 0), 0.75, "Random Forest ROC-AUC should exceed 0.75")
        self.assertGreater(rf.get("accuracy", 0), 0.70, "Random Forest accuracy should exceed 70%")

        # Check feature ranking presence
        features = data.get("top_feature_importances", [])
        self.assertGreater(len(features), 5)


if __name__ == "__main__":
    unittest.main()
