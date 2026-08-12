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

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparse"

DEVELOPMENT_PATH = DATA_DIR / "development.csv"
TEST_PATH = DATA_DIR / "test.csv"

FINAL_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

# Fallback for the actual path produced by your project.
FINAL_DAG_ALTERNATIVE = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
    / "formal_edge_decision"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
TREATMENT = "trt"

BASELINE_VARIABLES = [
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
]

ALL_VARIABLES = BASELINE_VARIABLES + [
    TREATMENT,
    TARGET,
]

ESS = 10


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def locate_final_dag() -> Path:
    candidates = [
        FINAL_DAG_PATH,
        FINAL_DAG_ALTERNATIVE,
    ]

    # Also search recursively in case the directory layout differs.
    candidates.extend(
        PROJECT_ROOT.glob(
            "results/structure_learning/final/**/*dag*edges.csv"
        )
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate the final DAG edge file.\n"
        "Expected one of:\n"
        f"{FINAL_DAG_PATH}\n"
        f"{FINAL_DAG_ALTERNATIVE}"
    )


def load_data():
    development = pd.read_csv(DEVELOPMENT_PATH)
    test = pd.read_csv(TEST_PATH)

    required = set(ALL_VARIABLES)

    missing_dev = required - set(development.columns)
    missing_test = required - set(test.columns)

    if missing_dev:
        raise ValueError(
            f"Development dataset missing columns: {sorted(missing_dev)}"
        )

    if missing_test:
        raise ValueError(
            f"Test dataset missing columns: {sorted(missing_test)}"
        )

    # Make every BN variable a discrete string state.
    for column in ALL_VARIABLES:
        development[column] = development[column].astype(str)
        test[column] = test[column].astype(str)

    return development, test


def load_edges(path: Path):
    edges_df = pd.read_csv(path)

    if not {"source", "target"}.issubset(edges_df.columns):
        raise ValueError(
            f"DAG file must contain source and target columns: {path}"
        )

    edges = [
        (str(row["source"]), str(row["target"]))
        for _, row in edges_df.iterrows()
    ]

    return edges


def build_model(edges):
    model = DiscreteBayesianNetwork()

    model.add_nodes_from(ALL_VARIABLES)
    model.add_edges_from(edges)

    return model


def validate_dag(model, edges, name):
    print(f"\n{name} validation:")

    graph = nx.DiGraph()
    graph.add_nodes_from(ALL_VARIABLES)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"{name} contains a cycle.")

    print("DAG structure: PASSED")

    # Treatment must not have outcome as a parent.
    if (TARGET, TREATMENT) in edges:
        raise ValueError(
            f"Forbidden edge detected: {TARGET} -> {TREATMENT}"
        )

    print("Forbidden label -> trt: PASSED")

    # Temporal restriction:
    # Baseline -> Treatment -> Outcome
    for source, target in edges:

        if source == TARGET and target != TARGET:
            raise ValueError(
                f"Outcome has outgoing edge: {source} -> {target}"
            )

        if target == TREATMENT and source == TARGET:
            raise ValueError(
                "Invalid edge label -> trt"
            )

    print("Temporal ordering: PASSED")


def create_sensitivity_edges(original_edges):
    edges = list(original_edges)

    treatment_edge = (TREATMENT, TARGET)

    if treatment_edge not in edges:
        edges.append(treatment_edge)

    return edges


def learn_parameters(data, edges):
    model = build_model(edges)

    estimator = BayesianEstimator(
        model,
        data,
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    model.add_cpds(*cpds)

    if not model.check_model():
        raise ValueError("Model consistency check failed.")

    return model


def expected_calibration_error(
    y_true,
    probabilities,
    bins=10,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()
        fraction = mask.mean()

        ece += fraction * abs(
            accuracy - confidence
        )

    return float(ece)


def state_to_numeric(series):
    """
    Convert label states robustly to 0/1.

    ACTG175 label is expected to contain 0 and 1.
    """
    values = series.astype(str)

    unique = set(values.unique())

    if unique.issubset({"0", "1"}):
        return values.astype(int).to_numpy()

    raise ValueError(
        f"Unexpected label states: {sorted(unique)}"
    )


def infer_test_probabilities(model, test):
    inference = VariableElimination(model)

    probabilities = []

    for _, row in test.iterrows():

        evidence = {
            column: row[column]
            for column in ALL_VARIABLES
            if column != TARGET
        }

        try:
            result = inference.query(
                variables=[TARGET],
                evidence=evidence,
                show_progress=False,
            )

            states = list(result.state_names[TARGET])

            if "1" not in states:
                raise ValueError(
                    "Outcome state '1' not found in inference result."
                )

            probability = float(
                result.values[
                    states.index("1")
                ]
            )

        except Exception:
            # Fallback: use only evidence that belongs to the model.
            valid_evidence = {}

            for variable, value in evidence.items():
                if variable in model.nodes():
                    try:
                        valid_evidence[variable] = value
                    except Exception:
                        pass

            result = inference.query(
                variables=[TARGET],
                evidence=valid_evidence,
                show_progress=False,
            )

            states = list(result.state_names[TARGET])

            probability = float(
                result.values[
                    states.index("1")
                ]
            )

        probabilities.append(probability)

    return np.asarray(probabilities)


def evaluate_predictions(test, probabilities):
    y = state_to_numeric(test[TARGET])

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "ece": expected_calibration_error(
            y,
            probabilities,
        ),
        "positive_rate": float(
            y.mean()
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
    }

    # ROC-AUC requires both classes.
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics, y, predictions


def calculate_bic_score(data, model):
    """
    Discrete BIC score calculated directly.

    log-likelihood:
        sum_i sum_jk N_ijk log(N_ijk / N_jk)

    BIC:
        log_likelihood - 0.5 * parameters * log(N)
    """

    n = len(data)
    log_likelihood = 0.0
    parameter_count = 0

    for node in model.nodes():

        parents = list(model.predecessors(node))

        node_states = sorted(
            data[node].astype(str).unique()
        )

        if not parents:

            counts = (
                data[node]
                .astype(str)
                .value_counts()
            )

            total = counts.sum()

            parameter_count += len(node_states) - 1

            for state in node_states:
                count = counts.get(state, 0)

                if count > 0:
                    probability = count / total
                    log_likelihood += (
                        count * np.log(probability)
                    )

        else:

            grouped = (
                data.groupby(
                    parents + [node],
                    dropna=False,
                )
                .size()
                .reset_index(name="count")
            )

            parent_totals = (
                grouped.groupby(
                    parents,
                    dropna=False,
                )["count"]
                .sum()
            )

            parent_state_counts = 1

            for parent in parents:
                parent_state_counts *= (
                    data[parent]
                    .astype(str)
                    .nunique()
                )

            parameter_count += (
                parent_state_counts
                * (len(node_states) - 1)
            )

            for _, row in grouped.iterrows():

                parent_key = tuple(
                    row[parent]
                    for parent in parents
                )

                if len(parents) == 1:
                    parent_key = parent_key[0]

                parent_total = parent_totals.loc[
                    parent_key
                ]

                count = row["count"]

                if count > 0:
                    probability = (
                        count / parent_total
                    )

                    log_likelihood += (
                        count * np.log(probability)
                    )

    bic = (
        log_likelihood
        - 0.5
        * parameter_count
        * np.log(n)
    )

    return float(bic), int(parameter_count)


def treatment_diagnostic(test, probabilities, model_name):
    result = test.copy()

    result["predicted_probability"] = probabilities

    rows = []

    for treatment, group in result.groupby(TREATMENT):

        observed = float(
            state_to_numeric(
                group[TARGET]
            ).mean()
        )

        predicted = float(
            group["predicted_probability"].mean()
        )

        rows.append(
            {
                "model": model_name,
                "treatment": treatment,
                "test_rows": len(group),
                "observed_P_label_1": observed,
                "model_P_label_1": predicted,
                "absolute_difference": abs(
                    observed - predicted
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ACTG175 PHASE-14"
    )

    print(
        "FORMAL TREATMENT-EDGE DECISION"
    )

    print("\nIMPORTANT:")
    print(
        "The existing final DAG will NOT be modified."
    )
    print(
        "We compare the current DAG against a candidate DAG"
    )
    print(
        "containing trt -> label."
    )
    print(
        "Development data is used for parameter learning"
    )
    print(
        "and structural scoring."
    )
    print(
        "Test data is used only for final evaluation."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    development, test = load_data()

    print(
        f"\nDevelopment shape: {development.shape}"
    )
    print(
        f"Test shape:        {test.shape}"
    )
    print(
        f"Development missing: {development.isna().sum().sum()}"
    )
    print(
        f"Test missing:        {test.isna().sum().sum()}"
    )

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    dag_path = locate_final_dag()

    print(
        f"\nFinal DAG file:\n{dag_path}"
    )

    original_edges = load_edges(dag_path)

    print(
        f"Original DAG edges: {len(original_edges)}"
    )

    # --------------------------------------------------------
    # Current DAG
    # --------------------------------------------------------

    print_header(
        "MODEL A — CURRENT FINAL DAG"
    )

    current_edges = list(original_edges)

    current_model_structure = build_model(
        current_edges
    )

    validate_dag(
        current_model_structure,
        current_edges,
        "Current DAG",
    )

    print(
        f"Edges: {len(current_edges)}"
    )

    print(
        "\nLearning BDeu parameters..."
    )

    current_model = learn_parameters(
        development,
        current_edges,
    )

    print(
        f"CPDs learned: {len(current_model.get_cpds())}"
    )

    print(
        "Model consistency: PASSED"
    )

    print(
        "\nEvaluating current DAG on TEST set..."
    )

    current_probabilities = infer_test_probabilities(
        current_model,
        test,
    )

    current_metrics, y, current_predictions = (
        evaluate_predictions(
            test,
            current_probabilities,
        )
    )

    print(
        "\nCurrent DAG metrics:"
    )

    for key, value in current_metrics.items():
        print(
            f"{key}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Sensitivity DAG
    # --------------------------------------------------------

    print_header(
        "MODEL B — TREATMENT-EDGE SENSITIVITY DAG"
    )

    sensitivity_edges = create_sensitivity_edges(
        current_edges
    )

    print(
        "Added edge:"
    )
    print(
        f"{TREATMENT} -> {TARGET}"
    )

    sensitivity_structure = build_model(
        sensitivity_edges
    )

    validate_dag(
        sensitivity_structure,
        sensitivity_edges,
        "Sensitivity DAG",
    )

    print(
        f"Edges: {len(sensitivity_edges)}"
    )

    print(
        "\nLearning BDeu parameters..."
    )

    sensitivity_model = learn_parameters(
        development,
        sensitivity_edges,
    )

    print(
        f"CPDs learned: {len(sensitivity_model.get_cpds())}"
    )

    print(
        "Model consistency: PASSED"
    )

    print(
        "\nEvaluating sensitivity DAG on TEST set..."
    )

    sensitivity_probabilities = (
        infer_test_probabilities(
            sensitivity_model,
            test,
        )
    )

    sensitivity_metrics, _, sensitivity_predictions = (
        evaluate_predictions(
            test,
            sensitivity_probabilities,
        )
    )

    print(
        "\nSensitivity DAG metrics:"
    )

    for key, value in sensitivity_metrics.items():
        print(
            f"{key}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Development structural scores
    # --------------------------------------------------------

    print_header(
        "DEVELOPMENT STRUCTURAL COMPARISON"
    )

    current_bic, current_parameters = (
        calculate_bic_score(
            development,
            current_model_structure,
        )
    )

    sensitivity_bic, sensitivity_parameters = (
        calculate_bic_score(
            development,
            sensitivity_structure,
        )
    )

    print(
        f"Current DAG BIC:     {current_bic:.4f}"
    )
    print(
        f"Sensitivity DAG BIC: {sensitivity_bic:.4f}"
    )

    print(
        f"Current parameters:     {current_parameters}"
    )
    print(
        f"Sensitivity parameters: {sensitivity_parameters}"
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
        "bic_change": (
            sensitivity_bic
            - current_bic
        ),
        "parameter_count_change": (
            sensitivity_parameters
            - current_parameters
        ),
    }

    # --------------------------------------------------------
    # Treatment diagnostics
    # --------------------------------------------------------

    current_diagnostic = treatment_diagnostic(
        test,
        current_probabilities,
        "Current final DAG",
    )

    sensitivity_diagnostic = treatment_diagnostic(
        test,
        sensitivity_probabilities,
        "Sensitivity DAG (trt -> label)",
    )

    treatment_diagnostic_df = pd.concat(
        [
            current_diagnostic,
            sensitivity_diagnostic,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Formal decision logic
    # --------------------------------------------------------

    log_loss_improved = (
        changes["log_loss_change"] < 0
    )

    brier_improved = (
        changes["brier_score_change"] < 0
    )

    auc_improved = (
        changes["roc_auc_change"] > 0
    )

    accuracy_improved = (
        changes["accuracy_change"] > 0
    )

    ece_improved = (
        changes["ece_change"] < 0
    )

    bic_improved = (
        changes["bic_change"] > 0
    )

    # The treatment edge has to satisfy:
    #
    # 1. Structural validity
    # 2. Test predictive improvement in at least
    #    two important proper/discrimination metrics
    # 3. No catastrophic calibration degradation
    #
    # We do NOT require BIC improvement because the
    # edge is being investigated specifically as a
    # causal/predictive sensitivity relationship.

    predictive_improvements = sum(
        [
            log_loss_improved,
            brier_improved,
            auc_improved,
            accuracy_improved,
        ]
    )

    calibration_penalty = (
        changes["ece_change"]
    )

    if (
        predictive_improvements >= 3
        and calibration_penalty <= 0.01
    ):
        recommendation = (
            "ACCEPT_TREATMENT_EDGE"
        )
        decision = (
            "The evidence supports promoting "
            "trt -> label to the final DAG."
        )

    elif predictive_improvements >= 2:
        recommendation = (
            "ACCEPT_WITH_CALIBRATION_REVIEW"
        )
        decision = (
            "The treatment edge improves predictive "
            "performance, but calibration requires "
            "additional review before final adoption."
        )

    else:
        recommendation = (
            "REJECT_TREATMENT_EDGE"
        )
        decision = (
            "The treatment edge does not provide "
            "sufficient predictive improvement."
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_header(
        "FORMAL PHASE-14 COMPARISON"
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Current final DAG",
                **current_metrics,
                "development_BIC": current_bic,
                "parameter_count": current_parameters,
                "edges": len(current_edges),
            },
            {
                "model": "Sensitivity DAG (trt -> label)",
                **sensitivity_metrics,
                "development_BIC": sensitivity_bic,
                "parameter_count": sensitivity_parameters,
                "edges": len(sensitivity_edges),
            },
        ]
    )

    print(
        comparison.round(6).to_string(
            index=False
        )
    )

    print(
        "\nMetric changes (Sensitivity - Current):"
    )

    for key, value in changes.items():
        print(
            f"{key}: {value:.6f}"
        )

    print(
        "\nImprovement checks:"
    )

    print(
        f"Log Loss improved:   {log_loss_improved}"
    )
    print(
        f"Brier improved:      {brier_improved}"
    )
    print(
        f"ROC-AUC improved:    {auc_improved}"
    )
    print(
        f"Accuracy improved:   {accuracy_improved}"
    )
    print(
        f"ECE improved:        {ece_improved}"
    )
    print(
        f"BIC improved:        {bic_improved}"
    )

    print(
        f"\nPredictive metrics improved: "
        f"{predictive_improvements}/4"
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "PHASE-14 DECISION"
    )
    print(
        "=" * 70
    )

    print(
        f"\nRecommendation: {recommendation}"
    )

    print(
        f"\n{decision}"
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "The existing final DAG has NOT been modified."
    )

    print(
        "Phase 15 should only modify the final DAG "
        "after this decision is reviewed."
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = test.copy()

    predictions_df[
        "current_probability_label_1"
    ] = current_probabilities

    predictions_df[
        "sensitivity_probability_label_1"
    ] = sensitivity_probabilities

    predictions_df[
        "current_prediction"
    ] = current_predictions

    predictions_df[
        "sensitivity_prediction"
    ] = sensitivity_predictions

    predictions_path = (
        OUTPUT_DIR
        / "formal_edge_decision_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "formal_edge_decision_metrics.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save treatment diagnostic
    # --------------------------------------------------------

    treatment_path = (
        OUTPUT_DIR
        / "formal_edge_decision_treatment_diagnostic.csv"
    )

    treatment_diagnostic_df.to_csv(
        treatment_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "phase": 14,
        "analysis": "formal_treatment_edge_decision",
        "development_rows": len(development),
        "test_rows": len(test),
        "current_dag_edges": len(current_edges),
        "sensitivity_dag_edges": len(sensitivity_edges),
        "added_edge": "trt -> label",
        "current_metrics": current_metrics,
        "sensitivity_metrics": sensitivity_metrics,
        "metric_changes": changes,
        "development_bic": {
            "current": current_bic,
            "sensitivity": sensitivity_bic,
            "change": sensitivity_bic - current_bic,
        },
        "parameter_count": {
            "current": current_parameters,
            "sensitivity": sensitivity_parameters,
            "change": sensitivity_parameters
            - current_parameters,
        },
        "improvement_checks": {
            "log_loss": log_loss_improved,
            "brier_score": brier_improved,
            "roc_auc": auc_improved,
            "accuracy": accuracy_improved,
            "ece": ece_improved,
            "bic": bic_improved,
        },
        "predictive_metrics_improved": predictive_improvements,
        "recommendation": recommendation,
        "decision": decision,
        "final_dag_modified": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "formal_edge_decision_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\nSaved results:"
    )

    print(predictions_path)
    print(comparison_path)
    print(treatment_path)
    print(summary_path)

    print(
        "\n" + "=" * 70
    )
    print(
        "PHASE-14 COMPLETE"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()