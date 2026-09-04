import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "continuous_model" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Exact continuous variables from ACTG175
CONTINUOUS_VARIABLES = [
    "age",      # Patient age in years
    "wtkg",     # Body weight in kilograms
    "karnof",   # Karnofsky Performance Scale (70-100)
    "preanti",  # Prior ART duration in days
    "cd40",     # Baseline CD4 count in cells/mm^3
    "cd80",     # Baseline CD8 count in cells/mm^3
]

CATEGORICAL_VARIABLES = [
    "hemo",     # Hemophilia (0/1)
    "homo",     # Homosexual activity (0/1)
    "drugs",    # History of IV drug use (0/1)
    "oprior",   # Prior opportunistic infection (0/1)
    "z30",      # Prior zidovudine use in 30 days before trial (0/1)
    "race",     # Race (0=White, 1=Non-white)
    "gender",   # Gender (0=Female, 1=Male)
    "strat",    # Antiretroviral history stratum (1=Naive, 2=Experienced, 3=Long-experienced)
    "symptom",  # Symptomatic HIV at baseline (0=No, 1=Yes)
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


class ContinuousCausalModel:
    """
    Continuous / Hybrid Structural Causal Model (SCM) with G-computation response surfaces.
    Preserves continuous clinical variables without arbitrary discretization,
    supports exact differentiable gradient sensitivities, E-values, and CATE treatment benefits.
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
        """
        Fits the continuous preprocessor and arm-specific response surfaces
        strictly on the development dataset (zero test leakage).
        """
        dev_df = dev_df.copy()

        # Ensure correct datatypes
        for col in CONTINUOUS_VARIABLES:
            dev_df[col] = dev_df[col].astype(float)
        for col in CATEGORICAL_VARIABLES:
            dev_df[col] = dev_df[col].astype(str)

        # Learn preprocessor on Development partition only
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

        # Extract StandardScaler parameters for exact analytical gradient sensitivity
        num_scaler = self.preprocessor.named_transformers_["num"]
        for idx, col in enumerate(CONTINUOUS_VARIABLES):
            self.scaler_means[col] = float(num_scaler.mean_[idx])
            self.scaler_stds[col] = float(num_scaler.scale_[idx])

        # Feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_VARIABLES)
        self.feature_names = list(CONTINUOUS_VARIABLES) + list(cat_feature_names)

        # Fit arm-specific response surfaces
        for t in [0, 1, 2, 3]:
            mask_t = (trt_dev == t)
            if np.sum(mask_t) == 0:
                raise ValueError(f"No development samples found for treatment arm {t}")

            clf = LogisticRegression(
                penalty="l2",
                C=self.c_reg,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_seed,
            )
            clf.fit(X_dev[mask_t], y_dev[mask_t])
            self.arm_models[t] = clf

        # Compute empirical distribution of Absolute Risk Reduction (ARR) across development cohort
        # ARR = P(Y=1 | do(0)) - min_t P(Y=1 | do(t))
        dev_arrs = []
        for i in range(len(dev_df)):
            p0 = self.arm_models[0].predict_proba(X_dev[i : i + 1])[0, 1]
            p_best = min(self.arm_models[t].predict_proba(X_dev[i : i + 1])[0, 1] for t in [1, 2, 3])
            dev_arrs.append(p0 - p_best)
        self.dev_risk_reductions = np.sort(np.array(dev_arrs))

        self.is_fitted = True
        return self

    def _transform_patient(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms patient inputs using development-fitted parameters."""
        if not self.is_fitted or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before running inference.")

        row_data = {}
        for col in CONTINUOUS_VARIABLES:
            val = patient_dict.get(col)
            if val is None or val == "":
                raise ValueError(f"Missing continuous variable: {col}")
            row_data[col] = [float(val)]

        for col in CATEGORICAL_VARIABLES:
            val = patient_dict.get(col)
            if val is None or val == "":
                raise ValueError(f"Missing categorical variable: {col}")
            row_data[col] = [str(val).strip()]

        df_row = pd.DataFrame(row_data)
        return self.preprocessor.transform(df_row[ALL_FEATURES])

    def predict_interventional_arm(
        self, patient_dict: Dict[str, Any], trt_val: int
    ) -> Tuple[float, float]:
        """Computes interventional probability under Pearlian do(trt = t)."""
        X_vec = self._transform_patient(patient_dict)
        probs = self.arm_models[trt_val].predict_proba(X_vec)[0]
        return float(probs[0]), float(probs[1])

    def analyze_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates causal interventions do(trt = k) across all 4 treatment arms,
        ranks them by expected utility, and computes:
        - Exact continuous gradient sensitivities (dP/dX_cont)
        - E-value for unmeasured confounding sensitivity
        - CATE treatment benefit percentile
        - What-If CD4 trajectory simulation points
        """
        X_vec = self._transform_patient(patient_dict)
        treatments = []

        for t in [0, 1, 2, 3]:
            probs = self.arm_models[t].predict_proba(X_vec)[0]
            p0 = float(probs[0])
            p1 = float(probs[1])
            eu = p0 * UTILITY_LABEL_0 + p1 * UTILITY_LABEL_1

            treatments.append({
                "treatment": t,
                "name": TREATMENTS[t]["name"],
                "short_name": TREATMENTS[t]["short_name"],
                "p_label_0": round(p0, 4),
                "p_label_1": round(p1, 4),
                "expected_utility": round(eu, 4),
            })

        treatments.sort(key=lambda x: x["expected_utility"], reverse=True)
        rank = 0
        prev_eu = None
        for item in treatments:
            if item["expected_utility"] != prev_eu:
                rank += 1
                prev_eu = item["expected_utility"]
            item["rank"] = rank

        treatments.sort(key=lambda x: x["treatment"])
        recommended = min(treatments, key=lambda x: (-x["expected_utility"], x["treatment"]))

        for item in treatments:
            item["is_recommended"] = bool(item["treatment"] == recommended["treatment"])

        sorted_ranking = sorted(treatments, key=lambda x: x["rank"])

        # Counterfactual risk comparison vs standard ZDV Monotherapy (Arm 0)
        monotherapy_arm = next((t for t in treatments if t["treatment"] == 0), treatments[0])
        p1_mono = max(float(monotherapy_arm["p_label_1"]), 1e-4)
        p1_rec = max(float(recommended["p_label_1"]), 1e-4)
        risk_delta_vs_monotherapy = float(monotherapy_arm["p_label_1"] - recommended["p_label_1"])

        # 1. E-Value Sensitivity Analysis for Unmeasured Confounding (VanderWeele & Ding)
        risk_ratio = p1_rec / p1_mono
        if risk_ratio < 1.0:
            rr_star = 1.0 / risk_ratio
            e_value = rr_star + math.sqrt(rr_star * (rr_star - 1.0))
        else:
            e_value = 1.0

        e_value_analysis = {
            "risk_ratio_vs_monotherapy": round(risk_ratio, 4),
            "e_value_point": round(e_value, 2),
            "interpretation": (
                f"An unmeasured confounder would need an association of at least RR = {e_value:.2f} "
                f"with both treatment assignment and clinical progression to explain away the observed benefit."
            ),
            "is_robust": bool(e_value >= 1.5),
        }

        # 2. CATE & Treatment Benefit Percentile
        arr = risk_delta_vs_monotherapy
        cate_percentile = (
            float(np.mean(self.dev_risk_reductions <= arr) * 100.0)
            if len(self.dev_risk_reductions) > 0
            else 50.0
        )
        treatment_benefit = {
            "absolute_risk_reduction": round(arr, 4),
            "arr_percentage_points": round(arr * 100, 2),
            "benefit_percentile": round(cate_percentile, 1),
            "benefit_tier": (
                "Exceptional (Top 10%)"
                if cate_percentile >= 90
                else ("High (Top 25%)" if cate_percentile >= 75 else ("Moderate" if cate_percentile >= 50 else "Mild"))
            ),
        }

        # 3. Exact continuous gradient sensitivity analysis under recommended treatment
        rec_model = self.arm_models[recommended["treatment"]]
        logistic_derivative_factor = p1_rec * (1.0 - p1_rec)

        continuous_sensitivities = []
        for idx, var in enumerate(CONTINUOUS_VARIABLES):
            coef_std = float(rec_model.coef_[0][idx])
            scale_std = self.scaler_stds[var]
            raw_derivative = (logistic_derivative_factor * coef_std) / scale_std

            # Meaningful clinical step unit
            step_unit = 50.0 if "cd" in var else (10.0 if var == "karnof" else (100.0 if var == "preanti" else 5.0))
            risk_change_per_step = raw_derivative * step_unit

            continuous_sensitivities.append({
                "variable": var,
                "raw_value": float(patient_dict[var]),
                "raw_derivative": round(raw_derivative, 6),
                "clinical_step_unit": step_unit,
                "risk_change_per_step": round(risk_change_per_step, 4),
                "percentage_points_per_step": round(risk_change_per_step * 100, 2),
                "direction": "increases_risk" if raw_derivative > 1e-5 else ("reduces_risk" if raw_derivative < -1e-5 else "neutral"),
            })

        continuous_sensitivities.sort(key=lambda x: abs(x["raw_derivative"]), reverse=True)

        # 4. Interactive What-If CD4 Trajectory Simulation (sweep CD4 from 50 to 750)
        cd4_trajectory = []
        current_cd4 = float(patient_dict.get("cd40", 350))
        for cd4_val in range(50, 801, 50):
            sim_dict = dict(patient_dict)
            sim_dict["cd40"] = float(cd4_val)
            try:
                X_sim = self._transform_patient(sim_dict)
                point = {"cd4": cd4_val}
                for arm_idx in [0, 1, 2, 3]:
                    point[f"arm_{arm_idx}"] = round(float(self.arm_models[arm_idx].predict_proba(X_sim)[0, 1]), 4)
                cd4_trajectory.append(point)
            except Exception:
                pass

        return {
            "model_type": "continuous_hybrid_scm",
            "model_name": "Model B: Continuous / Hybrid SCM (G-Computation)",
            "information_preservation": "Exact continuous numerical measurements preserved without discretization",
            "inputs": patient_dict,
            "treatments": treatments,
            "ranking": sorted_ranking,
            "recommended": {
                "treatment": recommended["treatment"],
                "name": recommended["name"],
                "short_name": recommended["short_name"],
                "p_label_0": recommended["p_label_0"],
                "p_label_1": recommended["p_label_1"],
                "expected_utility": recommended["expected_utility"],
                "risk_delta_vs_monotherapy": round(risk_delta_vs_monotherapy, 4),
                "e_value_analysis": e_value_analysis,
                "treatment_benefit": treatment_benefit,
                "continuous_sensitivities": continuous_sensitivities,
            },
            "what_if_trajectory": {
                "variable": "cd40",
                "label": "Baseline CD4 Count (cells/mm³)",
                "current_value": current_cd4,
                "trajectory_points": cd4_trajectory,
            },
            "utility_model": {
                "label_0_utility": UTILITY_LABEL_0,
                "label_1_utility": UTILITY_LABEL_1,
            },
        }

    def save(self, filepath: Path | None = None) -> Path:
        """Saves the fitted model artifact."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "continuous_causal_model.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @classmethod
    def load(cls, filepath: Path | None = None) -> "ContinuousCausalModel":
        """Loads a saved model artifact."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "continuous_causal_model.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
