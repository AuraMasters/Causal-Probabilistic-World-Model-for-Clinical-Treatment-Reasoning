import json
import math
import sys
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# REUSE THE EXISTING MODEL CODE
# ============================================================
# The final Bayesian Network, development-only BDeu parameter
# learning, development-fitted discretization, the 23-edge DAG
# and the do(trt) intervention logic all live in
# src/analysis/treatment/intervention_analysis.py and are
# imported here rather than re-implemented.
# ============================================================

from src.analysis.treatment.intervention_analysis import (
    NUMERICAL_VARIABLES,
    TREATMENTS,
    build_model,
    discretize_numerical_value,
    get_states,
    intervention_probability,
    load_data,
    load_discretization_metadata,
    load_edges,
)

# ============================================================
# UTILITY MODEL (from decision_analysis.py)
# ============================================================

UTILITY_LABEL_0 = 1.0
UTILITY_LABEL_1 = 0.0

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


# ============================================================
# SINGLETON STATE
# ============================================================

_STATE = {}


def get_data():
    if "data" not in _STATE:
        _STATE["data"] = load_data()
    return _STATE["data"]


def get_metadata():
    if "metadata" not in _STATE:
        _STATE["metadata"] = load_discretization_metadata()
    return _STATE["metadata"]


def get_edges():
    if "edges" not in _STATE:
        edges = load_edges()
        if len(edges) != 23:
            raise ValueError(
                f"Expected 23 final DAG edges, found {len(edges)}."
            )
        _STATE["edges"] = edges
    return _STATE["edges"]


def get_states_map():
    if "states" not in _STATE:
        _STATE["states"] = get_states(get_data())
    return _STATE["states"]


def get_model():
    if "model" not in _STATE:
        _STATE["model"] = build_model(
            get_data(),
            get_edges(),
        )
    return _STATE["model"]


# ============================================================
# VALIDATION
# ============================================================

def validate_numerical_input(variable, raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{variable} must be a number."
        )
    if not math.isfinite(value):
        raise ValueError(
            f"{variable} must be a finite number."
        )
    if variable == "preanti" and value < 0:
        raise ValueError(
            "Pre-ART exposure cannot be negative."
        )
    if value < 0:
        raise ValueError(
            f"{variable} cannot be negative."
        )
    return value


def validate_categorical_input(variable, raw, states):
    value = str(raw).strip()
    valid = states.get(variable, [])
    if value not in valid:
        raise ValueError(
            f"{variable} must be one of: {', '.join(valid)}."
        )
    return value


# ============================================================
# DISCRETIZATION FEEDBACK
# ============================================================

def build_discretization_feedback(variable, value, metadata):
    edges = metadata["variables"][variable]["edges"]

    ranges = []
    if variable == "preanti":
        ranges = [
            {"condition": "value == 0", "state": "zero"},
            {
                "condition": "0 < value <= " + _fmt(edges[1]),
                "state": "positive_1",
            },
            {
                "condition": (
                    _fmt(edges[1])
                    + " < value <= "
                    + _fmt(edges[2])
                ),
                "state": "positive_2",
            },
            {
                "condition": "value > " + _fmt(edges[2]),
                "state": "positive_3",
            },
        ]
    elif variable == "karnof":
        ranges = [
            {
                "condition": "value <= " + _fmt(edges[1]),
                "state": "karnof_1",
            },
            {
                "condition": "value > " + _fmt(edges[1]),
                "state": "karnof_2",
            },
        ]
    else:
        ranges = [
            {
                "condition": "value <= " + _fmt(edges[1]),
                "state": f"{variable}_1",
            },
            {
                "condition": (
                    _fmt(edges[1])
                    + " < value <= "
                    + _fmt(edges[2])
                ),
                "state": f"{variable}_2",
            },
            {
                "condition": "value > " + _fmt(edges[2]),
                "state": f"{variable}_3",
            },
        ]

    state = discretize_numerical_value(
        variable,
        value,
        metadata,
    )

    return {
        "variable": variable,
        "value": value,
        "state": state,
        "ranges": ranges,
    }


def _fmt(number):
    return _trim_float(number)


def _trim_float(number):
    value = float(number)
    if value == int(value):
        return str(int(value))
    return repr(value)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_patient(inputs):
    get_data()
    metadata = get_metadata()
    states = get_states_map()
    model = get_model()

    evidence = {}
    numerical_feedback = []

    for variable in NUMERICAL_VARIABLES:
        value = validate_numerical_input(
            variable,
            inputs.get(variable),
        )
        state = discretize_numerical_value(
            variable,
            value,
            metadata,
        )
        evidence[variable] = state
        numerical_feedback.append(
            build_discretization_feedback(
                variable,
                value,
                metadata,
            )
        )

    for variable in CATEGORICAL_VARIABLES:
        value = validate_categorical_input(
            variable,
            inputs.get(variable),
            states,
        )
        evidence[variable] = value

    treatments = []

    for treatment in ["0", "1", "2", "3"]:
        probability_0, probability_1 = (
            intervention_probability(
                model,
                evidence,
                treatment,
            )
        )

        expected_utility = (
            probability_0 * UTILITY_LABEL_0
            + probability_1 * UTILITY_LABEL_1
        )

        treatments.append(
            {
                "treatment": int(treatment),
                "name": TREATMENTS[treatment]["name"],
                "short_name": TREATMENTS[treatment]["short_name"],
                "p_label_0": probability_0,
                "p_label_1": probability_1,
                "expected_utility": expected_utility,
            }
        )

    treatments.sort(
        key=lambda row: row["expected_utility"],
        reverse=True,
    )

    rank = 0
    previous = None
    for row in treatments:
        if row["expected_utility"] != previous:
            rank += 1
            previous = row["expected_utility"]
        row["rank"] = rank

    treatments.sort(
        key=lambda row: row["treatment"]
    )

    recommended = min(
        treatments,
        key=lambda row: (
            -row["expected_utility"],
            row["treatment"],
        ),
    )

    for row in treatments:
        row["is_recommended"] = bool(
            row["treatment"] == recommended["treatment"]
        )

    sorted_treatments = sorted(
        treatments,
        key=lambda row: row["rank"],
    )

    return {
        "evidence": evidence,
        "numerical_feedback": numerical_feedback,
        "treatments": treatments,
        "ranking": sorted_treatments,
        "recommended": {
            "treatment": recommended["treatment"],
            "name": recommended["name"],
            "short_name": recommended["short_name"],
            "p_label_0": recommended["p_label_0"],
            "p_label_1": recommended["p_label_1"],
            "expected_utility": recommended["expected_utility"],
        },
        "utility_model": {
            "label_0_utility": UTILITY_LABEL_0,
            "label_1_utility": UTILITY_LABEL_1,
        },
        "decision_rule": (
            "Select the treatment with the highest predicted "
            "probability of Label 0 (equivalently the lowest "
            "predicted probability of Label 1, i.e. the highest "
            "expected utility)."
        ),
    }


# ============================================================
# OVERVIEW
# ============================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _json_edge(edge):
    try:
        value = float(edge)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def get_overview():
    data = get_data()
    edges = get_edges()

    final_metrics_path = (
        PROJECT_ROOT
        / "results"
        / "validation"
        / "final_model"
        / "metrics.json"
    )

    decision_metrics_path = (
        PROJECT_ROOT
        / "results"
        / "validation"
        / "treatment_decision"
        / "decision_validation_metrics.json"
    )

    final_metrics = read_json(final_metrics_path)
    decision_metrics = read_json(decision_metrics_path)

    decision_count = round(
        decision_metrics["recommended_better_than_observed_rate"]
        * decision_metrics["test_patients"]
    )

    discretization = deepcopy(get_metadata())

    for variable_metadata in discretization["variables"].values():
        variable_metadata["edges"] = [
            _json_edge(edge)
            for edge in variable_metadata["edges"]
        ]

    overview = {
        "dataset": {
            "name": "ACTG175",
            "patients": 2139,
            "development_rows": len(data),
            "test_rows": final_metrics["test_rows"],
            "treatments": len(TREATMENTS),
        },
        "model": {
            "dag_edges": len(edges),
            "nodes": len({
                node for edge in edges for node in edge
            }),
            "parameter_learning": "BDeu (ESS=10), development-only",
        },
        "dag": [
            {"source": source, "target": target}
            for source, target in edges
        ],
        "variables": {
            "numerical": NUMERICAL_VARIABLES,
            "categorical": CATEGORICAL_VARIABLES,
        },
        "states": get_states_map(),
        "discretization": discretization,
        "treatments": [
            {
                "treatment": int(key),
                "name": value["name"],
                "short_name": value["short_name"],
            }
            for key, value in sorted(TREATMENTS.items())
        ],
        "validation": {
            "log_loss": final_metrics["log_loss"],
            "brier_score": final_metrics["brier_score"],
            "roc_auc": final_metrics["roc_auc"],
            "accuracy": final_metrics["accuracy"],
            "ece": final_metrics["ece"],
            "test_patients": final_metrics["test_rows"],
        },
        "treatment_decision_validation": {
            "better_count": decision_count,
            "total": decision_metrics["test_patients"],
            "rate": decision_metrics[
                "recommended_better_than_observed_rate"
            ],
        },
        "utility_model": {
            "label_0_utility": UTILITY_LABEL_0,
            "label_1_utility": UTILITY_LABEL_1,
        },
    }

    return overview
