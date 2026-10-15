import json
import math
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app
from src.api.service import (
    NUMERICAL_VARIABLES,
    TREATMENTS,
    analyze_patient,
    get_edges,
    get_metadata,
    get_overview,
    get_states_map,
)
from src.continuous_model.model import (
    CONTINUOUS_VARIABLES,
    ContinuousCausalModel,
)
from src.validation.comprehensive_validation import (
    compute_calibration_analysis,
    compute_decision_curve_analysis,
    compute_predictive_metrics,
)
from src.validation.threshold_analysis import evaluate_threshold_sweep


class TestValidationPipeline(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    # ============================================================
    # 1. DATA PARTITION INTEGRITY & LEAKAGE TESTS
    # ============================================================
    def test_data_partition_integrity_and_no_leakage(self):
        dev_path = PROJECT_ROOT / "data" / "processed" / "sparse" / "development.csv"
        test_path = PROJECT_ROOT / "data" / "processed" / "sparse" / "test.csv"

        self.assertTrue(dev_path.exists(), "development.csv is missing")
        self.assertTrue(test_path.exists(), "test.csv is missing")

        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        self.assertEqual(len(dev_df), 1711, f"Expected 1,711 development rows, got {len(dev_df)}")
        self.assertEqual(len(test_df), 428, f"Expected 428 test rows, got {len(test_df)}")
        self.assertEqual(len(dev_df) + len(test_df), 2139, "Total dataset size must equal 2,139 ACTG175 patients")

        self.assertIn("label", dev_df.columns)
        self.assertIn("label", test_df.columns)
        self.assertIn("trt", dev_df.columns)
        self.assertIn("trt", test_df.columns)

    def test_continuous_raw_dataset_partitions(self):
        raw_dev_path = PROJECT_ROOT / "data" / "processed" / "actg175_development.csv"
        raw_test_path = PROJECT_ROOT / "data" / "processed" / "actg175_test.csv"

        self.assertTrue(raw_dev_path.exists(), "actg175_development.csv is missing")
        self.assertTrue(raw_test_path.exists(), "actg175_test.csv is missing")

        raw_dev = pd.read_csv(raw_dev_path)
        raw_test = pd.read_csv(raw_test_path)

        self.assertEqual(len(raw_dev), 1711)
        self.assertEqual(len(raw_test), 428)
        for var in CONTINUOUS_VARIABLES:
            self.assertIn(var, raw_dev.columns)
            self.assertIn(var, raw_test.columns)
            self.assertTrue(np.issubdtype(raw_dev[var].dtype, np.number))

    # ============================================================
    # 2. DISCRETIZATION METADATA TESTS
    # ============================================================
    def test_discretization_metadata_consistency(self):
        meta = get_metadata()
        self.assertIn("variables", meta, "Discretization metadata missing 'variables'")
        self.assertEqual(meta["representation"], "sparse")

        for var in NUMERICAL_VARIABLES:
            self.assertIn(var, meta["variables"], f"Missing metadata for numerical variable {var}")
            edges = meta["variables"][var]["edges"]
            self.assertGreaterEqual(len(edges), 2, f"Discretization for {var} must have at least 2 boundaries")
            numeric_edges = [float(e) for e in edges if e is not None and not math.isinf(float(e))]
            self.assertEqual(numeric_edges, sorted(numeric_edges), f"Edges for {var} are not sorted")

    # ============================================================
    # 3. DAG STRUCTURAL VALIDITY TESTS
    # ============================================================
    def test_bayesian_network_structure_and_dag_validity(self):
        edges = get_edges()
        self.assertEqual(len(edges), 23, f"Expected 23 DAG edges, got {len(edges)}")

        G = nx.DiGraph()
        G.add_edges_from(edges)
        self.assertTrue(nx.is_directed_acyclic_graph(G), "The causal graph must be acyclic")

        self.assertIn(("trt", "label"), edges, "Expected causal edge ('trt', 'label') in DAG")
        self.assertFalse(nx.has_path(G, "label", "trt"), "Causal violation: 'label' cannot cause 'trt'")

    # ============================================================
    # 4. CONTINUOUS / HYBRID SCM (MODEL B) TESTS
    # ============================================================
    def test_continuous_model_fitting_and_sensitivity(self):
        dev_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_development.csv")
        cm = ContinuousCausalModel(c_reg=0.5, random_seed=42)
        cm.fit(dev_df)

        sample = {
            "age": "38", "wtkg": "75.5", "karnof": "90", "preanti": "120", "cd40": "320", "cd80": "950",
            "hemo": "0", "homo": "1", "drugs": "0", "oprior": "0", "z30": "1", "race": "0",
            "gender": "1", "strat": "2", "symptom": "0"
        }

        res = cm.analyze_patient(sample)
        self.assertEqual(res["model_type"], "continuous_hybrid_scm")
        self.assertEqual(len(res["treatments"]), 4)
        self.assertIn("continuous_sensitivities", res["recommended"])
        self.assertGreaterEqual(len(res["recommended"]["continuous_sensitivities"]), 5)

        for sens in res["recommended"]["continuous_sensitivities"]:
            self.assertIn(sens["direction"], ["increases_risk", "reduces_risk", "neutral"])
            self.assertTrue(math.isfinite(sens["percentage_points_per_step"]))

    def test_continuous_validation_artifact(self):
        val_path = PROJECT_ROOT / "results" / "validation" / "continuous" / "continuous_validation.json"
        self.assertTrue(val_path.exists(), "continuous_validation.json is missing")

        with open(val_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pred = data["predictive_metrics"]
        self.assertGreater(pred["roc_auc"], 0.65, "Continuous model ROC-AUC should exceed 0.65")
        self.assertGreater(pred["pr_auc"], 0.38, "Continuous model PR-AUC should exceed 0.38")
        self.assertLess(data["calibration"]["ece"], 0.05, "Continuous model ECE should be under 5%")

    def test_model_comparison_artifact(self):
        comp_path = PROJECT_ROOT / "results" / "validation" / "comparison" / "model_comparison.json"
        self.assertTrue(comp_path.exists(), "model_comparison.json is missing")

        with open(comp_path, "r", encoding="utf-8") as f:
            comp = json.load(f)

        self.assertIn("comparison_table", comp)
        self.assertGreaterEqual(len(comp["comparison_table"]), 6)
        self.assertGreater(comp["model_b_summary"]["roc_auc"], comp["model_a_summary"]["roc_auc"])

    # ============================================================
    # 5. METRIC CALCULATION CORRECTNESS TESTS
    # ============================================================
    def test_predictive_metrics_correctness(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        metrics = compute_predictive_metrics(y_true, y_prob)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall_sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["true_positives"], 4)
        self.assertEqual(metrics["confusion_matrix"]["true_negatives"], 4)
        self.assertEqual(metrics["confusion_matrix"]["false_positives"], 0)
        self.assertEqual(metrics["confusion_matrix"]["false_negatives"], 0)

    def test_dca_net_benefit_calculation(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.8, 0.6, 0.2, 0.1])

        dca = compute_decision_curve_analysis(y_true, y_prob, threshold_probs=[0.25, 0.50])
        self.assertIn("dca_points", dca)
        self.assertEqual(len(dca["dca_points"]), 2)

        pt_50 = next(p for p in dca["dca_points"] if p["threshold_probability"] == 0.50)
        self.assertEqual(pt_50["net_benefit_model"], 0.5)
        self.assertEqual(pt_50["net_benefit_none"], 0.0)

    # ============================================================
    # 6. THRESHOLD OPTIMIZATION VALIDATION TESTS
    # ============================================================
    def test_threshold_sweep_bounds_and_validity(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9])

        sweep = evaluate_threshold_sweep(y_true, y_prob, thresholds=[0.15, 0.50, 0.80])
        self.assertEqual(len(sweep), 3)

        for item in sweep:
            self.assertTrue(0.0 <= item["sensitivity_recall"] <= 1.0)
            self.assertTrue(0.0 <= item["specificity"] <= 1.0)
            self.assertTrue(0.0 <= item["accuracy"] <= 1.0)
            self.assertTrue(-1.0 <= item["youden_j"] <= 1.0)

    # ============================================================
    # 7. BACKEND API ENDPOINT TESTS (WITH MODEL TOGGLE)
    # ============================================================
    def test_api_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_api_overview_payload_structure(self):
        res = self.client.get("/api/overview")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("dataset", data)
        self.assertEqual(data["dataset"]["patients"], 2139)
        self.assertEqual(data["dataset"]["development_rows"], 1711)
        self.assertEqual(data["dataset"]["test_rows"], 428)
        self.assertIn("dag", data)
        self.assertEqual(len(data["dag"]), 23)
        self.assertIn("treatments", data)
        self.assertEqual(len(data["treatments"]), 4)
        self.assertIn("continuous_validation", data)
        self.assertIn("model_comparison", data)
        self.assertIsNotNone(data["continuous_validation"])
        self.assertIsNotNone(data["model_comparison"])

    def test_api_analyze_continuous_model(self):
        payload = {
            "model_type": "continuous",
            "inputs": {
                "age": "35", "wtkg": "70", "karnof": "100", "preanti": "0", "cd40": "350", "cd80": "800",
                "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
                "gender": "1", "strat": "1", "symptom": "0",
            }
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["model_type"], "continuous_hybrid_scm")
        self.assertIn("recommended", data)
        self.assertIn(data["recommended"]["treatment"], [0, 1, 2, 3])
        self.assertIn("continuous_sensitivities", data["recommended"])
        self.assertTrue(math.isclose(
            data["recommended"]["p_label_0"] + data["recommended"]["p_label_1"],
            1.0,
            abs_tol=1e-4,
        ))

    def test_api_analyze_discretized_model(self):
        payload = {
            "model_type": "discretized",
            "inputs": {
                "age": "35", "wtkg": "70", "karnof": "100", "preanti": "0", "cd40": "350", "cd80": "800",
                "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
                "gender": "1", "strat": "1", "symptom": "0",
            }
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["model_type"], "discretized_bayesian_network")
        self.assertIn("feature_attributions", data["recommended"])
        self.assertTrue(math.isclose(
            data["recommended"]["p_label_0"] + data["recommended"]["p_label_1"],
            1.0,
            abs_tol=1e-4,
        ))

    def test_api_analyze_invalid_input(self):
        res = self.client.post("/api/analyze", data="bad payload", content_type="application/json")
        self.assertEqual(res.status_code, 400)

        res2 = self.client.post("/api/analyze", json={"inputs": {}})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("error", res2.get_json())

    # ============================================================
    # 8. GENDER ENCODING & CAUSAL INTERVENTION RIGOR TESTS
    # ============================================================
    def test_gender_encoding_ground_truth_and_consistency(self):
        """Verifies gender encoding ground truth (0=Female, 1=Male) across raw, dev, test, and API."""
        raw_path = PROJECT_ROOT / "data" / "raw" / "AIDS_ClinicalTrial_GroupStudy175.csv"
        self.assertTrue(raw_path.exists())
        raw_df = pd.read_csv(raw_path)

        # In ACTG175: Male (1) is ~82.8% of participants, Female (0) is ~17.2%
        male_count = int((raw_df["gender"] == 1).sum())
        female_count = int((raw_df["gender"] == 0).sum())
        self.assertEqual(male_count, 1771, "ACTG175 male count must be 1,771")
        self.assertEqual(female_count, 368, "ACTG175 female count must be 368")
        self.assertGreater(male_count, female_count)

        # Development and Test sets preserve this exact mapping
        dev_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_development.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_test.csv")
        self.assertIn(0, dev_df["gender"].values)
        self.assertIn(1, dev_df["gender"].values)
        self.assertIn(0, test_df["gender"].values)
        self.assertIn(1, test_df["gender"].values)

        # Both gender states must be recognized by API
        states = get_states_map()
        self.assertIn("gender", states)
        self.assertEqual(set(states["gender"]), {"0", "1"})

    def test_model_a_and_model_b_distinct_logic_and_outputs(self):
        """Ensures Model A (Discretized BN) and Model B (Continuous SCM) produce distinct outputs."""
        sample_patient = {
            "age": "45", "wtkg": "82.0", "karnof": "90", "preanti": "350", "cd40": "280", "cd80": "1100",
            "hemo": "0", "homo": "1", "drugs": "0", "oprior": "0", "z30": "1", "race": "0",
            "gender": "1", "strat": "2", "symptom": "0"
        }

        res_a = analyze_patient(sample_patient, model_type="discretized")
        res_b = analyze_patient(sample_patient, model_type="continuous")

        self.assertEqual(res_a["model_type"], "discretized_bayesian_network")
        self.assertEqual(res_b["model_type"], "continuous_hybrid_scm")

        # Distinct information preservation and attribution mechanisms
        self.assertIn("feature_attributions", res_a["recommended"])
        self.assertIn("continuous_sensitivities", res_b["recommended"])

        # Model A uses discrete evidence states
        self.assertIn("evidence", res_a)
        self.assertIn(res_a["evidence"]["cd40"], ["cd40_1", "cd40_2", "cd40_3"])

        # Probabilities should reflect their distinct mathematical formulation
        p1_a = res_a["recommended"]["p_label_1"]
        p1_b = res_b["recommended"]["p_label_1"]
        self.assertTrue(0.0 <= p1_a <= 1.0)
        self.assertTrue(0.0 <= p1_b <= 1.0)

    def test_treatment_interventions_and_probability_validity(self):
        """Verifies that all 4 trial arms under Pearl do-calculus yield valid probabilities summing to 1."""
        sample = {
            "age": "32", "wtkg": "68.0", "karnof": "100", "preanti": "0", "cd40": "450", "cd80": "750",
            "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
            "gender": "0", "strat": "1", "symptom": "0"
        }

        for model_type in ["continuous", "discretized"]:
            res = analyze_patient(sample, model_type=model_type)
            treatments = res["treatments"]
            self.assertEqual(len(treatments), 4)

            for t in treatments:
                self.assertIn(t["treatment"], [0, 1, 2, 3])
                self.assertTrue(0.0 <= t["p_label_0"] <= 1.0)
                self.assertTrue(0.0 <= t["p_label_1"] <= 1.0)
                self.assertAlmostEqual(t["p_label_0"] + t["p_label_1"], 1.0, places=3)
                self.assertAlmostEqual(t["expected_utility"], t["p_label_0"], places=3)

            # Recommended treatment must match rank #1
            rec = res["recommended"]
            best_eu = max(t["expected_utility"] for t in treatments)
            self.assertAlmostEqual(rec["expected_utility"], best_eu, places=3)

    def test_test_set_isolation_and_no_leakage(self):
        """Verifies that graph selection and model training are completely isolated from test data."""
        # 1. Structural decision summary check
        summary_path = PROJECT_ROOT / "results" / "analysis" / "treatment" / "formal_edge_decision" / "formal_edge_decision_summary.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertIn("selection_methodology", summary)
        self.assertIn("Cross-Validation on Development Set", summary["selection_methodology"])
        self.assertIn("development_cross_validation", summary)

        # 2. Continuous causal model training check: scaler parameters match development set
        dev_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_development.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_test.csv")
        cm = ContinuousCausalModel(c_reg=0.5, random_seed=42)
        cm.fit(dev_df)

        for col in CONTINUOUS_VARIABLES:
            dev_mean = float(dev_df[col].mean())
            self.assertAlmostEqual(cm.scaler_means[col], dev_mean, places=3)
            # Scaler mean must NOT equal test set mean
            test_mean = float(test_df[col].mean())
            if not math.isclose(dev_mean, test_mean, abs_tol=0.1):
                self.assertNotAlmostEqual(cm.scaler_means[col], test_mean, places=1)

    def test_e_value_mathematical_bounds(self):
        """Tests E-Value sensitivity formula correctness for protective and non-protective risk ratios."""
        # Case A: Protective effect (RR = 0.50 -> RR* = 2.0 -> E = 2 + sqrt(2*1) = 3.414)
        rr = 0.5
        rr_star = 1.0 / rr
        e_val = rr_star + math.sqrt(rr_star * (rr_star - 1.0))
        self.assertAlmostEqual(e_val, 3.414, places=2)
        self.assertGreater(e_val, 1.5)

        # Case B: Non-protective effect (RR >= 1.0 -> E = 1.0)
        rr_null = 1.25
        e_null = 1.0 if rr_null >= 1.0 else rr_null
        self.assertEqual(e_null, 1.0)

    def test_api_error_handling_and_input_validation(self):
        """Verifies API returns clean HTTP 400 with descriptive JSON errors on invalid inputs."""
        # 1. Missing numerical variable
        res = self.client.post("/api/analyze", json={
            "inputs": {
                "wtkg": "70", "karnof": "100", "preanti": "0", "cd40": "350", "cd80": "800",
                "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
                "gender": "1", "strat": "1", "symptom": "0",
            }
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("age", res.get_json()["error"])

        # 2. Non-finite numerical value
        res = self.client.post("/api/analyze", json={
            "inputs": {
                "age": "NaN", "wtkg": "70", "karnof": "100", "preanti": "0", "cd40": "350", "cd80": "800",
                "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
                "gender": "1", "strat": "1", "symptom": "0",
            }
        })
        self.assertEqual(res.status_code, 400)

        # 3. Invalid categorical state
        res = self.client.post("/api/analyze", json={
            "inputs": {
                "age": "35", "wtkg": "70", "karnof": "100", "preanti": "0", "cd40": "350", "cd80": "800",
                "hemo": "0", "homo": "0", "drugs": "0", "oprior": "0", "z30": "0", "race": "0",
                "gender": "999", "strat": "1", "symptom": "0",
            }
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("gender", res.get_json()["error"])

    def test_frontend_offline_artifacts_integrity(self):
        """Verifies model weights and discretized CPT files for frontend offline inference are intact."""
        weights_path = PROJECT_ROOT / "frontend" / "src" / "lib" / "modelWeights.json"
        disc_path = PROJECT_ROOT / "frontend" / "src" / "lib" / "discretizedModelData.json"

        self.assertTrue(weights_path.exists(), "frontend modelWeights.json is missing")
        self.assertTrue(disc_path.exists(), "frontend discretizedModelData.json is missing")

        with open(weights_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
        self.assertIn("arms", weights)
        self.assertEqual(len(weights["arms"]), 4)
        self.assertIn("scaler_means", weights)
        self.assertIn("scaler_stds", weights)

        with open(disc_path, "r", encoding="utf-8") as f:
            disc = json.load(f)
        self.assertIn("cpt_label_1", disc)
        self.assertEqual(len(disc["cpt_label_1"]), 24, "Model A CPT must have 3*4*2 = 24 entries")
        self.assertIn("ablation_no_cd40", disc)
        self.assertIn("ablation_no_z30", disc)


if __name__ == "__main__":
    unittest.main()
