import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ACTG175 Clinical Features
CONTINUOUS_VARIABLES = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
CATEGORICAL_VARIABLES = [
    "hemo",
    "homo",
    "drugs",
    "oprior",
    "z30",
    "race",
    "gender",
    "strat",
    "symptom",
]
ALL_FEATURES = CONTINUOUS_VARIABLES + CATEGORICAL_VARIABLES
TARGET = "label"
TREATMENT = "trt"

TREATMENTS = {
    0: {"name": "Zidovudine (ZDV)", "short_name": "ZDV"},
    1: {"name": "Zidovudine + Didanosine (ZDV + ddI)", "short_name": "ZDV + ddI"},
    2: {"name": "Zidovudine + Zalcitabine (ZDV + ddC)", "short_name": "ZDV + ddC"},
    3: {"name": "Didanosine (ddI)", "short_name": "ddI"},
}

UTILITY_LABEL_0 = 1.0
UTILITY_LABEL_1 = 0.0


def compute_e_value(p1_rec: float, p1_mono: float) -> Dict[str, Any]:
    """Formulates VanderWeele & Ding (2017) E-value for unmeasured confounding."""
    p1_mono = max(float(p1_mono), 1e-4)
    p1_rec = max(float(p1_rec), 1e-4)
    rr = p1_rec / p1_mono

    if rr < 1.0:
        rr_star = 1.0 / rr
        e_val = rr_star + math.sqrt(rr_star * (rr_star - 1.0))
    else:
        e_val = 1.0

    return {
        "risk_ratio_vs_monotherapy": round(rr, 4),
        "e_value_point": round(e_val, 2),
        "interpretation": (
            f"An unmeasured confounder would need an association of at least RR = {e_val:.2f} "
            f"with both treatment choice and disease progression to nullify the observed risk reduction."
        ),
        "is_robust": bool(e_val >= 1.5),
    }


class ContinuousCausalModel:
    """
    Model B: Continuous / Hybrid SCM with G-computation Response Surfaces.
    Preserves numerical biomarkers without binning, supports exact analytical derivatives,
    E-values, CATE treatment benefit scoring, and What-If trajectory simulation.
    """

    def __init__(self, c_reg: float = 0.5, random_seed: int = 42):
        self.c_reg = c_reg
        self.random_seed = random_seed
        self.preprocessor: ColumnTransformer | None = None
        self.arm_models: Dict[int, LogisticRegression] = {}
        self.scaler_means: Dict[str, float] = {}
        self.scaler_stds: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.dev_risk_reductions: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, dev_df: pd.DataFrame) -> "ContinuousCausalModel":
        dev_df = dev_df.copy()

        for col in CONTINUOUS_VARIABLES:
            dev_df[col] = dev_df[col].astype(float)
        for col in CATEGORICAL_VARIABLES:
            dev_df[col] = dev_df[col].astype(str)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), CONTINUOUS_VARIABLES),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_VARIABLES),
            ],
            remainder="drop",
        )

        X_dev = self.preprocessor.fit_transform(dev_df[ALL_FEATURES])
        y_dev = dev_df[TARGET].astype(int).to_numpy()
        trt_dev = dev_df[TREATMENT].astype(int).to_numpy()

        num_scaler = self.preprocessor.named_transformers_["num"]
        for idx, col in enumerate(CONTINUOUS_VARIABLES):
            self.scaler_means[col] = float(num_scaler.mean_[idx])
            self.scaler_stds[col] = float(num_scaler.scale_[idx])

        self.feature_names = list(self.preprocessor.get_feature_names_out())

        for t in [0, 1, 2, 3]:
            mask_t = trt_dev == t
            clf = LogisticRegression(
                C=self.c_reg,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_seed,
            )
            clf.fit(X_dev[mask_t], y_dev[mask_t])
            self.arm_models[t] = clf

        p_mono = self.arm_models[0].predict_proba(X_dev)[:, 1]
        p_opt = np.minimum.reduce([self.arm_models[t].predict_proba(X_dev)[:, 1] for t in [1, 2, 3]])
        self.dev_risk_reductions = np.sort(np.maximum(0.0, p_mono - p_opt))

        self.is_fitted = True
        return self

    def predict_arm_probabilities(self, X_input: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        if not self.is_fitted or self.preprocessor is None:
            raise RuntimeError("Model is not fitted yet.")

        X_vec = self.preprocessor.transform(X_input)
        results = {}
        for t in [0, 1, 2, 3]:
            probs = self.arm_models[t].predict_proba(X_vec)[0]
            p0 = float(probs[0])
            p1 = float(probs[1])
            results[t] = (p0, p1)
        return results

    def compute_analytical_gradients(self, X_input: pd.DataFrame, rec_arm: int) -> List[Dict[str, Any]]:
        X_vec = self.preprocessor.transform(X_input)
        clf = self.arm_models[rec_arm]
        p1 = float(clf.predict_proba(X_vec)[0][1])
        variance_factor = p1 * (1.0 - p1)

        sensitivities = []
        step_units = {"cd40": 50.0, "cd80": 100.0, "karnof": 10.0, "preanti": 100.0, "age": 5.0, "wtkg": 5.0}

        for var in CONTINUOUS_VARIABLES:
            feat_name = f"num__{var}"
            if feat_name in self.feature_names:
                feat_idx = self.feature_names.index(feat_name)
                beta_std = float(clf.coef_[0][feat_idx])
                sigma = self.scaler_stds.get(var, 1.0)
                raw_deriv = variance_factor * (beta_std / sigma)
                step = step_units.get(var, 1.0)
                risk_change = raw_deriv * step

                sensitivities.append({
                    "variable": var,
                    "raw_value": float(X_input[var].iloc[0]),
                    "raw_derivative": round(raw_deriv, 6),
                    "clinical_step_unit": step,
                    "risk_change_per_step": round(risk_change, 4),
                    "percentage_points_per_step": round(risk_change * 100, 2),
                    "direction": "increases_risk" if risk_change > 0.005 else ("reduces_risk" if risk_change < -0.005 else "neutral"),
                })

        sensitivities.sort(key=lambda s: abs(s["risk_change_per_step"]), reverse=True)
        return sensitivities

    def compute_what_if_trajectory(self, base_inputs: Dict[str, Any]) -> Dict[str, Any]:
        grid_cd4 = np.linspace(50, 750, 29)
        rows = []
        for cd4_val in grid_cd4:
            row = dict(base_inputs)
            row["cd40"] = float(cd4_val)
            rows.append(row)

        df_grid = pd.DataFrame(rows)
        for col in CONTINUOUS_VARIABLES:
            df_grid[col] = df_grid[col].astype(float)
        for col in CATEGORICAL_VARIABLES:
            df_grid[col] = df_grid[col].astype(str)

        X_grid = self.preprocessor.transform(df_grid[ALL_FEATURES])
        trajectories = []
        for idx, cd4_val in enumerate(grid_cd4):
            item = {"cd4": int(cd4_val)}
            for t in [0, 1, 2, 3]:
                item[f"arm_{t}"] = round(float(self.arm_models[t].predict_proba(X_grid[idx : idx + 1])[0][1]), 4)
            trajectories.append(item)

        return {
            "biomarker": "cd40",
            "current_value": float(base_inputs["cd40"]),
            "trajectory_points": trajectories,
        }

    def compute_cate_benefit(self, patient_p1_rec: float, patient_p1_mono: float) -> Dict[str, Any]:
        arr = max(0.0, float(patient_p1_mono - patient_p1_rec))
        if len(self.dev_risk_reductions) == 0:
            percentile = 50.0
        else:
            percentile = float(np.searchsorted(self.dev_risk_reductions, arr) / len(self.dev_risk_reductions) * 100.0)

        tier = "High Benefit" if percentile >= 75.0 else ("Moderate Benefit" if percentile >= 40.0 else "Low Benefit")
        return {
            "absolute_risk_reduction": round(arr, 4),
            "benefit_percentile": round(percentile, 1),
            "benefit_tier": tier,
        }

    def analyze_patient(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df_patient = pd.DataFrame([inputs])
        for col in CONTINUOUS_VARIABLES:
            df_patient[col] = df_patient[col].astype(float)
        for col in CATEGORICAL_VARIABLES:
            df_patient[col] = df_patient[col].astype(str)

        arm_probs = self.predict_arm_probabilities(df_patient[ALL_FEATURES])

        treatments = []
        for t in [0, 1, 2, 3]:
            p0, p1 = arm_probs[t]
            eu = p0 * UTILITY_LABEL_0 + p1 * UTILITY_LABEL_1
            treatments.append({
                "treatment": t,
                "name": TREATMENTS[t]["name"],
                "short_name": TREATMENTS[t]["short_name"],
                "p_label_0": round(p0, 4),
                "p_label_1": round(p1, 4),
                "expected_utility": round(eu, 4),
            })

        treatments.sort(key=lambda row: row["expected_utility"], reverse=True)
        rank = 0
        prev_eu = None
        for row in treatments:
            if row["expected_utility"] != prev_eu:
                rank += 1
                prev_eu = row["expected_utility"]
            row["rank"] = rank

        treatments.sort(key=lambda row: row["treatment"])
        recommended = min(treatments, key=lambda row: (-row["expected_utility"], row["treatment"]))

        for row in treatments:
            row["is_recommended"] = bool(row["treatment"] == recommended["treatment"])

        sorted_treatments = sorted(treatments, key=lambda row: row["rank"])
        monotherapy_arm = treatments[0]
        risk_delta = float(monotherapy_arm["p_label_1"] - recommended["p_label_1"])

        e_value = compute_e_value(recommended["p_label_1"], monotherapy_arm["p_label_1"])
        gradients = self.compute_analytical_gradients(df_patient, recommended["treatment"])
        what_if = self.compute_what_if_trajectory(inputs)
        benefit = self.compute_cate_benefit(recommended["p_label_1"], monotherapy_arm["p_label_1"])

        return {
            "model_type": "continuous_scm",
            "model_name": "Model B: Continuous / Hybrid SCM (Primary)",
            "information_preservation": "Exact continuous biomarkers (No binning loss)",
            "treatments": treatments,
            "ranking": sorted_treatments,
            "recommended": {
                "treatment": recommended["treatment"],
                "name": recommended["name"],
                "short_name": recommended["short_name"],
                "p_label_0": recommended["p_label_0"],
                "p_label_1": recommended["p_label_1"],
                "expected_utility": recommended["expected_utility"],
                "risk_delta_vs_monotherapy": round(risk_delta, 4),
                "treatment_benefit": benefit,
                "e_value_analysis": e_value,
                "continuous_sensitivities": gradients,
            },
            "what_if_trajectory": what_if,
            "utility_model": {"label_0_utility": UTILITY_LABEL_0, "label_1_utility": UTILITY_LABEL_1},
        }

    def save(self, filepath: Path | None = None) -> Path:
        if filepath is None:
            filepath = ARTIFACTS_DIR / "continuous_causal_model.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @classmethod
    def load(cls, filepath: Path | str | None = None) -> "ContinuousCausalModel":
        if filepath is None:
            filepath = ARTIFACTS_DIR / "continuous_causal_model.pkl"
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        class _ModelUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module.startswith("src.continuous_model") or module.startswith("src."):
                    module = "backend.models"
                return super().find_class(module, name)

        with open(path, "rb") as f:
            model = _ModelUnpickler(f).load()

        for clf in getattr(model, "arm_models", {}).values():
            if not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
        return model


class DiscretizedBayesianNetwork:
    """
    Model A: Baseline Discretized Bayesian Network with 23 learned DAG edges,
    BDeu parameter estimation, and exact Pearlian do-calculus inference via Variable Elimination.
    """

    def __init__(self, data: pd.DataFrame, edges: List[Tuple[str, str]], ess: int = 10):
        self.edges = edges
        self.ess = ess
        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(ALL_FEATURES + [TARGET, TREATMENT])
        self.model.add_edges_from(edges)

        estimator = BayesianEstimator(self.model, data)
        cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=ess)
        self.model.add_cpds(*cpds)

        if not self.model.check_model():
            raise ValueError("Bayesian Network consistency check failed.")

    def evaluate_intervention(self, evidence: Dict[str, str], treatment: int) -> Tuple[float, float]:
        """Evaluates P(label | evidence, do(trt=k)) under Pearl graph mutilation."""
        mutilated_model = self.model.do([TREATMENT], inplace=False)
        infer = VariableElimination(mutilated_model)

        query_evidence = dict(evidence)
        query_evidence[TREATMENT] = str(treatment)

        res = infer.query(variables=[TARGET], evidence=query_evidence, show_progress=False)
        states = res.state_names[TARGET]
        probs = res.values

        p0, p1 = 0.0, 0.0
        for state, val in zip(states, probs):
            if str(state) == "0":
                p0 = float(val)
            elif str(state) == "1":
                p1 = float(val)

        total = p0 + p1
        if total > 0:
            p0 /= total
            p1 /= total
        return p0, p1

    def analyze_patient(self, evidence: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        treatments = []
        for t in [0, 1, 2, 3]:
            p0, p1 = self.evaluate_intervention(evidence, t)
            eu = p0 * UTILITY_LABEL_0 + p1 * UTILITY_LABEL_1
            treatments.append({
                "treatment": t,
                "name": TREATMENTS[t]["name"],
                "short_name": TREATMENTS[t]["short_name"],
                "p_label_0": round(p0, 4),
                "p_label_1": round(p1, 4),
                "expected_utility": round(eu, 4),
            })

        treatments.sort(key=lambda r: r["expected_utility"], reverse=True)
        rank = 0
        prev_eu = None
        for r in treatments:
            if r["expected_utility"] != prev_eu:
                rank += 1
                prev_eu = r["expected_utility"]
            r["rank"] = rank

        treatments.sort(key=lambda r: r["treatment"])
        recommended = min(treatments, key=lambda r: (-r["expected_utility"], r["treatment"]))

        for r in treatments:
            r["is_recommended"] = bool(r["treatment"] == recommended["treatment"])

        sorted_treatments = sorted(treatments, key=lambda r: r["rank"])
        monotherapy_arm = treatments[0]
        risk_delta = float(monotherapy_arm["p_label_1"] - recommended["p_label_1"])

        e_value = compute_e_value(recommended["p_label_1"], monotherapy_arm["p_label_1"])

        # Bayesian evidence ablation sensitivities
        attributions = []
        for var in ALL_FEATURES:
            if var in evidence:
                ablated = {k: v for k, v in evidence.items() if k != var}
                _, ablated_p1 = self.evaluate_intervention(ablated, recommended["treatment"])
                delta_p = float(recommended["p_label_1"] - ablated_p1)
                attributions.append({
                    "feature": var,
                    "observed_state": evidence[var],
                    "risk_impact": round(delta_p, 4),
                    "direction": "increases_risk" if delta_p > 0.0005 else ("reduces_risk" if delta_p < -0.0005 else "neutral"),
                    "percentage_points": round(delta_p * 100, 2),
                })

        attributions.sort(key=lambda x: abs(x["risk_impact"]), reverse=True)

        return {
            "model_type": "discretized_bayesian_network",
            "model_name": "Model A: Discretized Bayesian Network (Baseline)",
            "information_preservation": "3-bin quantile discretization (baseline)",
            "evidence": evidence,
            "treatments": treatments,
            "ranking": sorted_treatments,
            "recommended": {
                "treatment": recommended["treatment"],
                "name": recommended["name"],
                "short_name": recommended["short_name"],
                "p_label_0": recommended["p_label_0"],
                "p_label_1": recommended["p_label_1"],
                "expected_utility": recommended["expected_utility"],
                "risk_delta_vs_monotherapy": round(risk_delta, 4),
                "e_value_analysis": e_value,
                "feature_attributions": attributions,
            },
            "utility_model": {"label_0_utility": UTILITY_LABEL_0, "label_1_utility": UTILITY_LABEL_1},
        }
