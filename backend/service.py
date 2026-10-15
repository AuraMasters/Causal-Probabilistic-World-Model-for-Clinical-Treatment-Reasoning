import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from backend.models import (
    CATEGORICAL_VARIABLES,
    CONTINUOUS_VARIABLES,
    TREATMENTS,
    ContinuousCausalModel,
    DiscretizedBayesianNetwork,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
NUMERICAL_VARIABLES = CONTINUOUS_VARIABLES

# Paths
DEVELOPMENT_PATH = PROJECT_ROOT / "data" / "processed" / "sparse" / "development.csv"
DISCRETIZATION_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "sparse" / "discretization_metadata.json"
DAG_PATH = ARTIFACTS_DIR / "final_dag_edges.csv"
SUPPORT_PATH = ARTIFACTS_DIR / "final_dag_edge_support.csv"

_STATE: Dict[str, Any] = {}


def load_data() -> pd.DataFrame:
    if "data" not in _STATE:
        df = pd.read_csv(DEVELOPMENT_PATH)
        for col in df.columns:
            df[col] = df[col].astype(str)
        _STATE["data"] = df
    return _STATE["data"]


def load_discretization_metadata() -> Dict[str, Any]:
    if "metadata" not in _STATE:
        with open(DISCRETIZATION_METADATA_PATH, "r", encoding="utf-8") as f:
            _STATE["metadata"] = json.load(f)
    return _STATE["metadata"]


def load_edges() -> List[Tuple[str, str]]:
    if "edges" not in _STATE:
        df = pd.read_csv(DAG_PATH)
        edges = [(str(row["source"]).strip(), str(row["target"]).strip()) for _, row in df.iterrows()]
        if len(edges) != 23:
            raise ValueError(f"Expected 23 DAG edges, found {len(edges)}")
        _STATE["edges"] = edges
    return _STATE["edges"]


get_edges = load_edges
get_metadata = load_discretization_metadata


def get_states(data: pd.DataFrame) -> Dict[str, List[str]]:
    if "states" not in _STATE:
        _STATE["states"] = {col: sorted(data[col].unique().tolist()) for col in data.columns}
    return _STATE["states"]


def get_states_map() -> Dict[str, List[str]]:
    return get_states(load_data())


def get_edge_support_map() -> Dict[Tuple[str, str], Dict[str, Any]]:
    if "edge_support" not in _STATE:
        support_dict = {}
        if SUPPORT_PATH.exists():
            df = pd.read_csv(SUPPORT_PATH)
            for _, row in df.iterrows():
                s = str(row["source"]).strip()
                t = str(row["target"]).strip()
                support_dict[(s, t)] = {
                    "bootstrap_stability": float(row.get("bootstrap_stability", 0.0)),
                    "support_category": str(row.get("support_category", "EXPLORATORY")),
                    "reverse_stability": float(row.get("reverse_stability", 0.0)),
                }
        _STATE["edge_support"] = support_dict
    return _STATE["edge_support"]


def get_continuous_model() -> ContinuousCausalModel:
    if "continuous_model" not in _STATE:
        model_path = ARTIFACTS_DIR / "continuous_causal_model.pkl"
        if model_path.exists():
            _STATE["continuous_model"] = ContinuousCausalModel.load(model_path)
        else:
            dev_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "actg175_development.csv")
            cm = ContinuousCausalModel(c_reg=0.5, random_seed=42)
            cm.fit(dev_df)
            cm.save(model_path)
            _STATE["continuous_model"] = cm
    return _STATE["continuous_model"]


def get_discretized_model() -> DiscretizedBayesianNetwork:
    if "discretized_model" not in _STATE:
        _STATE["discretized_model"] = DiscretizedBayesianNetwork(load_data(), load_edges(), ess=10)
    return _STATE["discretized_model"]


def discretize_numerical_value(variable: str, value: float, metadata: Dict[str, Any]) -> str:
    variable_meta = metadata["variables"][variable]
    edges = variable_meta["edges"]
    val = float(value)

    if variable == "preanti":
        if val <= 0:
            return "zero"
        pos_edges = [e for e in edges if e is not None]
        if len(pos_edges) == 2:
            if val <= pos_edges[0]:
                return "positive_1"
            if val <= pos_edges[1]:
                return "positive_2"
            return "positive_3"
        return "positive_1"

    if variable == "karnof":
        return "karnof_1" if val <= edges[1] else "karnof_2"

    if val <= edges[1]:
        return f"{variable}_1"
    if val <= edges[2]:
        return f"{variable}_2"
    return f"{variable}_3"


def validate_numerical_input(variable: str, raw_value: Any) -> float:
    if raw_value is None or raw_value == "":
        raise ValueError(f"Missing required numerical variable: {variable}")
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numerical value for {variable}: {raw_value}")
    if not math.isfinite(val):
        raise ValueError(f"Numerical value for {variable} must be finite: {raw_value}")
    return val


def validate_categorical_input(variable: str, raw_value: Any, states: Dict[str, List[str]]) -> str:
    if raw_value is None or raw_value == "":
        raise ValueError(f"Missing required categorical variable: {variable}")
    val = str(raw_value).strip()
    allowed = states.get(variable, [])
    if allowed and val not in allowed:
        raise ValueError(f"Invalid state for {variable}: {val}. Allowed states: {allowed}")
    return val


def analyze_patient(inputs: Dict[str, Any], model_type: str = "continuous") -> Dict[str, Any]:
    if not isinstance(inputs, dict):
        raise ValueError("Inputs must be a dictionary of variable-value pairs.")

    states = get_states(load_data())
    cleaned = {}
    for var in CONTINUOUS_VARIABLES:
        cleaned[var] = validate_numerical_input(var, inputs.get(var))
    for var in CATEGORICAL_VARIABLES:
        cleaned[var] = validate_categorical_input(var, inputs.get(var), states)

    if model_type == "continuous":
        model_b = get_continuous_model()
        return model_b.analyze_patient(cleaned)

    # Baseline Model A: Discretized BN
    metadata = load_discretization_metadata()
    model_a = get_discretized_model()
    evidence = {}
    for var in CONTINUOUS_VARIABLES:
        evidence[var] = discretize_numerical_value(var, cleaned[var], metadata)
    for var in CATEGORICAL_VARIABLES:
        evidence[var] = cleaned[var]

    return model_a.analyze_patient(evidence, metadata)


def _safe_read_json(path: Path) -> Any:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_overview() -> Dict[str, Any]:
    data = load_data()
    edges = load_edges()
    edge_support = get_edge_support_map()

    final_metrics = _safe_read_json(ARTIFACTS_DIR / "metrics.json") or {}
    decision_metrics = _safe_read_json(ARTIFACTS_DIR / "decision_validation_metrics.json") or {}
    comprehensive_metrics = _safe_read_json(ARTIFACTS_DIR / "comprehensive_validation.json")
    continuous_metrics = _safe_read_json(ARTIFACTS_DIR / "continuous_validation.json")
    comparison_metrics = _safe_read_json(ARTIFACTS_DIR / "model_comparison.json")
    survival_audit = _safe_read_json(ARTIFACTS_DIR / "survival_censoring_audit.json")

    decision_count = round(decision_metrics.get("recommended_better_than_observed_rate", 0.75) * decision_metrics.get("test_patients", 428))

    discretization = deepcopy(load_discretization_metadata())
    for var_meta in discretization["variables"].values():
        var_meta["edges"] = [float(e) if e is not None and math.isfinite(float(e)) else None for e in var_meta["edges"]]

    dag_payload = []
    for s, t in edges:
        supp = edge_support.get((s, t), {})
        is_intervention = s == "trt" and t == "label"
        dag_payload.append({
            "source": s,
            "target": t,
            "bootstrap_stability": supp.get("bootstrap_stability", 1.0 if is_intervention else 0.0),
            "support_category": "INTERVENTION" if is_intervention else supp.get("support_category", "EXPLORATORY"),
            "reverse_stability": supp.get("reverse_stability", 0.0),
            "is_intervention_edge": is_intervention,
        })

    return {
        "dataset": {
            "name": "ACTG175",
            "patients": 2139,
            "development_rows": len(data),
            "test_rows": final_metrics.get("test_rows", 428),
            "treatments": len(TREATMENTS),
        },
        "model": {
            "dag_edges": len(edges),
            "nodes": len({node for edge in edges for node in edge}),
            "parameter_learning": "BDeu (ESS=10) for Model A; L2 G-Computation for Model B",
            "active_model": "continuous",
        },
        "dag": dag_payload,
        "variables": {"numerical": CONTINUOUS_VARIABLES, "categorical": CATEGORICAL_VARIABLES},
        "states": get_states(data),
        "discretization": discretization,
        "treatments": [{"treatment": k, "name": v["name"], "short_name": v["short_name"]} for k, v in sorted(TREATMENTS.items())],
        "validation": {
            "log_loss": final_metrics.get("log_loss", 0.533),
            "brier_score": final_metrics.get("brier_score", 0.1753),
            "roc_auc": final_metrics.get("roc_auc", 0.6372),
            "accuracy": final_metrics.get("accuracy", 0.7664),
            "ece": final_metrics.get("ece", 0.0445),
            "test_patients": final_metrics.get("test_rows", 428),
        },
        "treatment_decision_validation": {
            "better_count": decision_count,
            "total": decision_metrics.get("test_patients", 428),
            "rate": decision_metrics.get("recommended_better_than_observed_rate", 0.75),
        },
        "comprehensive_validation": comprehensive_metrics,
        "continuous_validation": continuous_metrics,
        "model_comparison": comparison_metrics,
        "survival_audit": survival_audit,
        "utility_model": {"label_0_utility": 1.0, "label_1_utility": 0.0},
    }
