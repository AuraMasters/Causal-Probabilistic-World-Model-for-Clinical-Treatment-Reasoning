from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "analysis" / "treatment"

DEVELOPMENT_PATH = (
    PROCESSED_DIR / "sparse" / "development.csv"
)

TEST_PATH = (
    PROCESSED_DIR / "sparse" / "test.csv"
)

FINAL_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

# Fallback for the earlier nested directory structure.
FINAL_DAG_FALLBACK = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag"
    / "final_dag_edges.csv"
)

ESS = 10

TARGET = "label"
TREATMENT = "trt"

RANDOM_STATE = 42

EXPECTED_COLUMNS = [
    "age",
    "wtkg",
    "hemo",
    "homo",
    "drugs",
    "karnof",
    "oprior",
    "z30",
    "preanti",
    "race",
    "gender",
    "strat",
    "symptom",
    "cd40",
    "cd80",
    "trt",
    "label",
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# PATH DISCOVERY
# ============================================================

def find_final_dag():
    candidates = [
        FINAL_DAG_PATH,
        FINAL_DAG_FALLBACK,
    ]

    # Also search recursively if the exact location changed.
    candidates.extend(
        PROJECT_ROOT.glob(
            "results/structure_learning/**/final_dag_edges.csv"
        )
    )

    seen = set()

    for path in candidates:
        path = Path(path)

        if str(path) in seen:
            continue

        seen.add(str(path))

        if path.exists():
            return path

    raise FileNotFoundError(
        "\nFinal DAG file could not be found.\n"
        "Expected one of:\n"
        f"  {FINAL_DAG_PATH}\n"
        f"  {FINAL_DAG_FALLBACK}\n"
    )


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df, name):

    required = set(EXPECTED_COLUMNS)

    missing_columns = sorted(
        required - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: "
            f"{missing_columns}"
        )

    if df[EXPECTED_COLUMNS].isnull().sum().sum() != 0:
        raise ValueError(
            f"{name} contains missing values."
        )

    print(
        f"{name} shape: {df.shape}"
    )

    print(
        f"{name} missing: "
        f"{int(df.isnull().sum().sum())}"
    )


# ============================================================
# DISCRETE STATE CONVERSION
# ============================================================

def convert_to_discrete(df):
    """
    Convert every variable to string states.

    This is important because the project uses a discrete
    Bayesian network and the sparse representation contains
    discretized states such as age_1, age_2, etc.
    """

    result = df.copy()

    for column in result.columns:
        result[column] = result[column].astype(str)

    return result


# ============================================================
# LOAD DAG
# ============================================================

def load_dag_edges(path):

    edges_df = pd.read_csv(path)

    if not {"source", "target"}.issubset(
        edges_df.columns
    ):
        raise ValueError(
            "DAG edge file must contain "
            "'source' and 'target' columns."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        edges.append(
            (source, target)
        )

    return edges


# ============================================================
# GRAPH VALIDATION
# ============================================================

def validate_dag(
    edges,
    variables,
    allow_treatment_edge=False,
):

    graph = nx.DiGraph()

    graph.add_nodes_from(variables)
    graph.add_edges_from(edges)

    # --------------------------------------------------------
    # DAG validation
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "Graph is not a DAG."
        )

    # --------------------------------------------------------
    # Forbidden edge
    # --------------------------------------------------------

    if ("label", "trt") in edges:
        raise ValueError(
            "Forbidden edge label -> trt detected."
        )

    # --------------------------------------------------------
    # Treatment edge
    # --------------------------------------------------------

    if not allow_treatment_edge:

        if ("trt", "label") in edges:
            raise ValueError(
                "trt -> label is not allowed in "
                "the current final DAG."
            )

    return graph


# ============================================================
# MODEL CREATION
# ============================================================

def build_model(
    edges,
    variables,
    allow_treatment_edge=False,
):

    validate_dag(
        edges=edges,
        variables=variables,
        allow_treatment_edge=allow_treatment_edge,
    )

    model = DiscreteBayesianNetwork(
        edges
    )

    # Make sure isolated variables such as trt are retained.
    model.add_nodes_from(variables)

    return model


# ============================================================
# PARAMETER LEARNING
# ============================================================

def learn_bdeu_parameters(
    model,
    development,
):

    print(
        "Learning BDeu parameters..."
    )

    # pgmpy 1.1.2 compatible API.
    #
    # Do NOT call:
    #
    # model.fit(
    #     data,
    #     prior_type="BDeu",
    #     equivalent_sample_size=10
    # )
    #
    # Those arguments are not accepted by
    # DiscreteBayesianNetwork.fit() in this version.

    estimator = BayesianEstimator(
        model,
        development,
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    model.add_cpds(*cpds)

    print(
        f"CPDs learned: {len(cpds)}"
    )

    # pgmpy model consistency check.
    model.check_model()

    print(
        "Model consistency: PASSED"
    )

    return model


# ============================================================
# INFERENCE
# ============================================================

def get_positive_probability(
    inference,
    evidence,
):

    query = inference.query(
        variables=[TARGET],
        evidence=evidence,
        show_progress=False,
    )

    # State order comes from the CPD.
    states = list(
        query.state_names[TARGET]
    )

    values = np.asarray(
        query.values,
        dtype=float,
    )

    if "1" not in states:
        raise ValueError(
            "Target state '1' not found in "
            f"inference result: {states}"
        )

    index = states.index("1")

    return float(
        values[index]
    )


# ============================================================
# PREDICTION GENERATION
# ============================================================

def generate_predictions(
    model,
    test,
):

    inference = VariableElimination(
        model
    )

    probabilities = []

    feature_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column != TARGET
    ]

    total = len(test)

    print(
        f"Patients to evaluate: {total}"
    )

    for i, (_, row) in enumerate(
        test.iterrows(),
        start=1,
    ):

        evidence = {
            column: row[column]
            for column in feature_columns
        }

        probability = (
            get_positive_probability(
                inference,
                evidence,
            )
        )

        probabilities.append(
            probability
        )

        if (
            i % 100 == 0
            or i == total
        ):
            print(
                f"Processed {i}/{total}"
            )

    return np.asarray(
        probabilities,
        dtype=float,
    )


# ============================================================
# CALIBRATION
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    bins=10,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    total = len(y_true)

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        confidence = probabilities[
            mask
        ].mean()

        accuracy = y_true[
            mask
        ].mean()

        ece += (
            count / total
        ) * abs(
            confidence - accuracy
        )

    return float(ece)


# ============================================================
# MODEL METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    probabilities = np.clip(
        probabilities,
        1e-12,
        1 - 1e-12,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "ece": float(
            expected_calibration_error(
                y_true,
                probabilities,
            )
        ),
        "positive_rate": float(
            np.mean(y_true)
        ),
        "mean_predicted_probability": float(
            np.mean(probabilities)
        ),
    }


# ============================================================
# TREATMENT DIAGNOSTIC
# ============================================================

def treatment_diagnostic(
    test,
    probabilities,
):

    rows = []

    for treatment in sorted(
        test[TREATMENT]
        .astype(str)
        .unique(),
        key=lambda x: int(x),
    ):

        mask = (
            test[TREATMENT].astype(str)
            == treatment
        )

        observed = (
            test.loc[
                mask,
                TARGET,
            ]
            .astype(int)
            .mean()
        )

        predicted = (
            probabilities[mask.to_numpy()]
            .mean()
        )

        rows.append(
            {
                "treatment": int(treatment),
                "test_rows": int(mask.sum()),
                "observed_P_label_1": float(
                    observed
                ),
                "model_P_label_1": float(
                    predicted
                ),
                "absolute_difference": float(
                    abs(
                        observed
                        - predicted
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EDGE DIFFERENCE
# ============================================================

def compare_edges(
    current_edges,
    sensitivity_edges,
):

    current_set = set(
        current_edges
    )

    sensitivity_set = set(
        sensitivity_edges
    )

    return {
        "current_edge_count": len(
            current_set
        ),
        "sensitivity_edge_count": len(
            sensitivity_set
        ),
        "added_edges": sorted(
            sensitivity_set
            - current_set
        ),
        "removed_edges": sorted(
            current_set
            - sensitivity_set
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    development,
    test,
    current_edges,
    sensitivity_edges,
    current_probabilities,
    sensitivity_probabilities,
    current_metrics,
    sensitivity_metrics,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = test.copy()

    predictions[
        "current_model_P_label_1"
    ] = current_probabilities

    predictions[
        "sensitivity_model_P_label_1"
    ] = sensitivity_probabilities

    predictions[
        "current_model_prediction"
    ] = (
        current_probabilities >= 0.5
    ).astype(int)

    predictions[
        "sensitivity_model_prediction"
    ] = (
        sensitivity_probabilities >= 0.5
    ).astype(int)

    predictions_path = (
        RESULTS_DIR
        / "edge_sensitivity_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "model": "current_final_dag",
                **current_metrics,
            },
            {
                "model": "sensitivity_trt_to_label",
                **sensitivity_metrics,
            },
        ]
    )

    metrics_path = (
        RESULTS_DIR
        / "edge_sensitivity_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Treatment diagnostic
    # --------------------------------------------------------

    current_treatment = (
        treatment_diagnostic(
            test,
            current_probabilities,
        )
    )

    sensitivity_treatment = (
        treatment_diagnostic(
            test,
            sensitivity_probabilities,
        )
    )

    current_treatment[
        "model"
    ] = "current_final_dag"

    sensitivity_treatment[
        "model"
    ] = "sensitivity_trt_to_label"

    treatment_df = pd.concat(
        [
            current_treatment,
            sensitivity_treatment,
        ],
        ignore_index=True,
    )

    treatment_path = (
        RESULTS_DIR
        / "edge_sensitivity_treatment_diagnostic.csv"
    )

    treatment_df.to_csv(
        treatment_path,
        index=False,
    )

    # --------------------------------------------------------
    # Edge comparison
    # --------------------------------------------------------

    edge_comparison = compare_edges(
        current_edges,
        sensitivity_edges,
    )

    edge_rows = []

    for source, target in sorted(
        set(current_edges)
        | set(sensitivity_edges)
    ):

        edge_rows.append(
            {
                "source": source,
                "target": target,
                "current_dag": (
                    (source, target)
                    in set(current_edges)
                ),
                "sensitivity_dag": (
                    (source, target)
                    in set(sensitivity_edges)
                ),
            }
        )

    edge_df = pd.DataFrame(
        edge_rows
    )

    edge_path = (
        RESULTS_DIR
        / "edge_comparison.csv"
    )

    edge_df.to_csv(
        edge_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metric changes
    # --------------------------------------------------------

    changes = {
        "log_loss_change": (
            sensitivity_metrics["log_loss"]
            - current_metrics["log_loss"]
        ),
        "brier_score_change": (
            sensitivity_metrics["brier_score"]
            - current_metrics["brier_score"]
        ),
        "roc_auc_change": (
            sensitivity_metrics["roc_auc"]
            - current_metrics["roc_auc"]
        ),
        "accuracy_change": (
            sensitivity_metrics["accuracy"]
            - current_metrics["accuracy"]
        ),
        "ece_change": (
            sensitivity_metrics["ece"]
            - current_metrics["ece"]
        ),
    }

    # --------------------------------------------------------
    # Final JSON summary
    # --------------------------------------------------------

    summary = {
        "phase": "Phase-13",
        "analysis": (
            "Treatment edge sensitivity analysis"
        ),
        "project_root": str(
            PROJECT_ROOT
        ),
        "development_path": str(
            DEVELOPMENT_PATH
        ),
        "test_path": str(
            TEST_PATH
        ),
        "final_dag_path": str(
            find_final_dag()
        ),
        "development_rows": int(
            len(development)
        ),
        "test_rows": int(
            len(test)
        ),
        "equivalent_sample_size": ESS,
        "current_model": {
            "name": "current_final_dag",
            "edges": [
                {
                    "source": source,
                    "target": target,
                }
                for source, target
                in current_edges
            ],
            "metrics": current_metrics,
        },
        "sensitivity_model": {
            "name": (
                "current_final_dag_plus_trt_to_label"
            ),
            "edges": [
                {
                    "source": source,
                    "target": target,
                }
                for source, target
                in sensitivity_edges
            ],
            "metrics": sensitivity_metrics,
        },
        "edge_comparison": edge_comparison,
        "metric_changes": changes,
        "final_dag_modified": False,
    }

    summary_path = (
        RESULTS_DIR
        / "edge_sensitivity_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("Saved results:")
    print(predictions_path)
    print(metrics_path)
    print(treatment_path)
    print(edge_path)
    print(summary_path)


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print_header(
        "ACTG175 PHASE-13\n"
        "TREATMENT EDGE SENSITIVITY ANALYSIS"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "The existing final DAG is NOT modified."
    )

    print(
        "The current DAG is compared against "
        "a sensitivity DAG containing trt -> label."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DEVELOPMENT_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_PATH}"
        )

    dag_path = find_final_dag()

    development_raw = pd.read_csv(
        DEVELOPMENT_PATH
    )

    test_raw = pd.read_csv(
        TEST_PATH
    )

    validate_dataset(
        development_raw,
        "Development",
    )

    validate_dataset(
        test_raw,
        "Test",
    )

    # --------------------------------------------------------
    # Convert to discrete states
    # --------------------------------------------------------

    development = convert_to_discrete(
        development_raw[
            EXPECTED_COLUMNS
        ]
    )

    test = convert_to_discrete(
        test_raw[
            EXPECTED_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    current_edges = load_dag_edges(
        dag_path
    )

    print()
    print(
        f"Original final DAG edges: "
        f"{len(current_edges)}"
    )

    variables = EXPECTED_COLUMNS

    # --------------------------------------------------------
    # Validate current DAG
    # --------------------------------------------------------

    print_section(
        "MODEL A — CURRENT FINAL DAG"
    )

    current_graph = validate_dag(
        current_edges,
        variables,
        allow_treatment_edge=False,
    )

    print(
        f"Edges: {len(current_edges)}"
    )

    print(
        "DAG validation: PASSED"
    )

    if ("label", "trt") in current_edges:
        raise ValueError(
            "Current final DAG contains "
            "forbidden label -> trt."
        )

    print(
        "Forbidden label -> trt: PASSED"
    )

    # --------------------------------------------------------
    # Build current model
    # --------------------------------------------------------

    current_model = build_model(
        current_edges,
        variables,
        allow_treatment_edge=False,
    )

    current_model = learn_bdeu_parameters(
        current_model,
        development,
    )

    current_inference = VariableElimination(
        current_model
    )

    print(
        "Inference engine: READY"
    )

    # --------------------------------------------------------
    # Current model predictions
    # --------------------------------------------------------

    print_section(
        "MODEL A — TEST SET EVALUATION"
    )

    current_probabilities = (
        generate_predictions(
            current_model,
            test,
        )
    )

    y_test = (
        test[TARGET]
        .astype(int)
        .to_numpy()
    )

    current_metrics = calculate_metrics(
        y_test,
        current_probabilities,
    )

    print()
    print(
        "Current final DAG metrics:"
    )

    for key, value in current_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Construct sensitivity DAG
    # --------------------------------------------------------

    print_section(
        "MODEL B — SENSITIVITY DAG"
    )

    sensitivity_edges = list(
        current_edges
    )

    treatment_edge = (
        TREATMENT,
        TARGET,
    )

    if treatment_edge not in sensitivity_edges:

        sensitivity_edges.append(
            treatment_edge
        )

    print(
        "Added sensitivity edge:"
    )

    print(
        "trt -> label"
    )

    # --------------------------------------------------------
    # Validate sensitivity DAG
    # --------------------------------------------------------

    sensitivity_graph = validate_dag(
        sensitivity_edges,
        variables,
        allow_treatment_edge=True,
    )

    print(
        f"Edges: "
        f"{len(sensitivity_edges)}"
    )

    print(
        "DAG validation: PASSED"
    )

    # --------------------------------------------------------
    # Build sensitivity model
    # --------------------------------------------------------

    sensitivity_model = build_model(
        sensitivity_edges,
        variables,
        allow_treatment_edge=True,
    )

    sensitivity_model = (
        learn_bdeu_parameters(
            sensitivity_model,
            development,
        )
    )

    sensitivity_inference = (
        VariableElimination(
            sensitivity_model
        )
    )

    print(
        "Inference engine: READY"
    )

    # --------------------------------------------------------
    # Sensitivity predictions
    # --------------------------------------------------------

    print_section(
        "MODEL B — TEST SET EVALUATION"
    )

    sensitivity_probabilities = (
        generate_predictions(
            sensitivity_model,
            test,
        )
    )

    sensitivity_metrics = (
        calculate_metrics(
            y_test,
            sensitivity_probabilities,
        )
    )

    print()
    print(
        "Sensitivity DAG metrics:"
    )

    for key, value in (
        sensitivity_metrics.items()
    ):

        print(
            f"{key}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print_section(
        "MODEL COMPARISON"
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Current final DAG",
                **current_metrics,
            },
            {
                "model": (
                    "Sensitivity DAG "
                    "(trt -> label)"
                ),
                **sensitivity_metrics,
            },
        ]
    )

    print(
        comparison.round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Metric changes
    # --------------------------------------------------------

    log_loss_change = (
        sensitivity_metrics["log_loss"]
        - current_metrics["log_loss"]
    )

    brier_change = (
        sensitivity_metrics["brier_score"]
        - current_metrics["brier_score"]
    )

    auc_change = (
        sensitivity_metrics["roc_auc"]
        - current_metrics["roc_auc"]
    )

    accuracy_change = (
        sensitivity_metrics["accuracy"]
        - current_metrics["accuracy"]
    )

    ece_change = (
        sensitivity_metrics["ece"]
        - current_metrics["ece"]
    )

    print()
    print(
        "Metric changes "
        "(Sensitivity - Current):"
    )

    print(
        f"Log Loss: "
        f"{log_loss_change:+.6f}"
    )

    print(
        f"Brier Score: "
        f"{brier_change:+.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{auc_change:+.6f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_change:+.6f}"
    )

    print(
        f"ECE: "
        f"{ece_change:+.6f}"
    )

    # --------------------------------------------------------
    # Treatment diagnostic
    # --------------------------------------------------------

    print_section(
        "TREATMENT-SPECIFIC TEST DIAGNOSTIC"
    )

    current_treatment = (
        treatment_diagnostic(
            test,
            current_probabilities,
        )
    )

    current_treatment[
        "model"
    ] = "Current final DAG"

    sensitivity_treatment = (
        treatment_diagnostic(
            test,
            sensitivity_probabilities,
        )
    )

    sensitivity_treatment[
        "model"
    ] = (
        "Sensitivity DAG (trt -> label)"
    )

    treatment_comparison = pd.concat(
        [
            current_treatment,
            sensitivity_treatment,
        ],
        ignore_index=True,
    )

    print(
        treatment_comparison.round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print_section(
        "INTERPRETATION"
    )

    better_log_loss = (
        sensitivity_metrics["log_loss"]
        < current_metrics["log_loss"]
    )

    better_brier = (
        sensitivity_metrics["brier_score"]
        < current_metrics["brier_score"]
    )

    better_auc = (
        sensitivity_metrics["roc_auc"]
        > current_metrics["roc_auc"]
    )

    better_ece = (
        sensitivity_metrics["ece"]
        < current_metrics["ece"]
    )

    improvements = sum(
        [
            better_log_loss,
            better_brier,
            better_auc,
            better_ece,
        ]
    )

    if improvements >= 3:

        recommendation = (
            "STRONGER_EVIDENCE_FOR_TREATMENT_EDGE"
        )

    elif improvements == 2:

        recommendation = (
            "MIXED_BUT_SUPPORTIVE_EVIDENCE"
        )

    elif improvements == 1:

        recommendation = (
            "WEAK_EVIDENCE_FOR_TREATMENT_EDGE"
        )

    else:

        recommendation = (
            "NO_PREDICTIVE_EVIDENCE_FOR_TREATMENT_EDGE"
        )

    print(
        f"Log Loss improved: "
        f"{better_log_loss}"
    )

    print(
        f"Brier Score improved: "
        f"{better_brier}"
    )

    print(
        f"ROC-AUC improved: "
        f"{better_auc}"
    )

    print(
        f"ECE improved: "
        f"{better_ece}"
    )

    print()
    print(
        f"Recommendation: "
        f"{recommendation}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a sensitivity analysis."
    )

    print(
        "The final DAG has NOT been modified."
    )

    print(
        "Adding trt -> label requires "
        "structural, causal, statistical, "
        "and domain justification."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        development=development,
        test=test,
        current_edges=current_edges,
        sensitivity_edges=sensitivity_edges,
        current_probabilities=current_probabilities,
        sensitivity_probabilities=sensitivity_probabilities,
        current_metrics=current_metrics,
        sensitivity_metrics=sensitivity_metrics,
    )

    print()
    print_header(
        "PHASE-13 COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()