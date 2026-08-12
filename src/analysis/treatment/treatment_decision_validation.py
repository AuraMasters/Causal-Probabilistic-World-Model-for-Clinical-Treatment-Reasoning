from pathlib import Path
import json
import warnings

import pandas as pd
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# ACTG175 PHASE-20
# TREATMENT DECISION VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "test.csv"
)

DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_model"
    / "dag"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "treatment_decision"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

ESS = 10

VARIABLES = [
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

TARGET = "label"
TREATMENT = "trt"


# ============================================================
# ACTG175 TREATMENT DEFINITIONS
# ============================================================

TREATMENTS = {
    "0": {
        "name": "Zidovudine (ZDV/AZT)",
        "short_name": "ZDV",
    },
    "1": {
        "name": "Zidovudine + Didanosine",
        "short_name": "ZDV + ddI",
    },
    "2": {
        "name": "Zidovudine + Zalcitabine",
        "short_name": "ZDV + ddC",
    },
    "3": {
        "name": "Didanosine (ddI)",
        "short_name": "ddI",
    },
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found:\n{path}"
        )

    data = pd.read_csv(path)

    missing_columns = [
        column
        for column in VARIABLES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} dataset is missing columns:\n"
            + ", ".join(missing_columns)
        )

    for column in VARIABLES:
        data[column] = data[column].astype(str)

    return data


# ============================================================
# LOAD FINAL DAG
# ============================================================

def load_edges():

    if not DAG_PATH.exists():
        raise FileNotFoundError(
            f"Final DAG not found:\n{DAG_PATH}"
        )

    edges_df = pd.read_csv(DAG_PATH)

    required_columns = {
        "source",
        "target",
    }

    if not required_columns.issubset(
        edges_df.columns
    ):
        raise ValueError(
            "DAG file must contain "
            "'source' and 'target' columns."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target"]
        ).strip()

        edges.append(
            (source, target)
        )

    return edges


# ============================================================
# VALIDATE DAG
# ============================================================

def validate_dag(
    model,
    edges,
):

    if len(edges) != 23:
        raise ValueError(
            f"Expected 23 DAG edges, "
            f"found {len(edges)}."
        )

    # Use NetworkX because the installed pgmpy
    # version does not expose model.is_dag().
    if not nx.is_directed_acyclic_graph(
        model
    ):
        raise ValueError(
            "Final DAG is not acyclic."
        )

    if (
        "label",
        "trt",
    ) in edges:

        raise ValueError(
            "Forbidden edge label -> trt detected."
        )

    if (
        "trt",
        "label",
    ) not in edges:

        raise ValueError(
            "Required edge trt -> label "
            "not found."
        )


# ============================================================
# BUILD BAYESIAN NETWORK
# ============================================================

def build_model(
    development,
    edges,
):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        VARIABLES
    )

    model.add_edges_from(
        edges
    )

    validate_dag(
        model,
        edges,
    )

    estimator = BayesianEstimator(
        model,
        development,
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    model.add_cpds(
        *cpds
    )

    if not model.check_model():
        raise ValueError(
            "Bayesian Network consistency check failed."
        )

    return model


# ============================================================
# TEST-STATE VALIDATION
# ============================================================

def validate_test_states(
    development,
    test,
):

    for variable in VARIABLES:

        development_states = set(
            development[
                variable
            ].unique()
        )

        test_states = set(
            test[
                variable
            ].unique()
        )

        unseen_states = (
            test_states
            - development_states
        )

        if unseen_states:

            raise ValueError(
                f"Test contains unseen states "
                f"for {variable}: "
                f"{sorted(unseen_states)}"
            )


# ============================================================
# INTERVENTIONAL PROBABILITY
# ============================================================

def get_intervention_probability(
    model,
    patient,
    treatment,
):

    # Remove incoming edges to treatment.
    # The treatment is then fixed by intervention.
    intervention_model = model.do(
        [TREATMENT],
        inplace=False,
    )

    inference = VariableElimination(
        intervention_model
    )

    evidence = {}

    # Patient characteristics only.
    # Treatment itself is supplied through
    # the intervention.
    for variable in VARIABLES:

        if variable in {
            TARGET,
            TREATMENT,
        }:
            continue

        evidence[
            variable
        ] = str(
            patient[variable]
        )

    # Set the intervention.
    evidence[
        TREATMENT
    ] = str(treatment)

    result = inference.query(
        variables=[TARGET],
        evidence=evidence,
        show_progress=False,
    )

    states = result.state_names[
        TARGET
    ]

    probabilities = result.values

    probability_0 = 0.0
    probability_1 = 0.0

    for state, probability in zip(
        states,
        probabilities,
    ):

        if str(state) == "0":

            probability_0 = float(
                probability
            )

        elif str(state) == "1":

            probability_1 = float(
                probability
            )

    return (
        probability_0,
        probability_1,
    )


# ============================================================
# EVALUATE ONE PATIENT
# ============================================================

def evaluate_patient(
    model,
    patient,
):

    treatment_rows = []

    for treatment in [
        "0",
        "1",
        "2",
        "3",
    ]:

        (
            probability_0,
            probability_1,
        ) = get_intervention_probability(
            model,
            patient,
            treatment,
        )

        treatment_rows.append(
            {
                "treatment": int(
                    treatment
                ),
                "treatment_name":
                    TREATMENTS[
                        treatment
                    ]["name"],
                "short_name":
                    TREATMENTS[
                        treatment
                    ]["short_name"],
                "p_label_0":
                    probability_0,
                "p_label_1":
                    probability_1,
            }
        )

    results = pd.DataFrame(
        treatment_rows
    )

    # Lowest predicted P(label=1)
    # is the current treatment-selection rule.
    best_index = results[
        "p_label_1"
    ].idxmin()

    best_treatment = int(
        results.loc[
            best_index,
            "treatment",
        ]
    )

    observed_treatment = int(
        patient[TREATMENT]
    )

    best_probability = float(
        results.loc[
            best_index,
            "p_label_1",
        ]
    )

    observed_probability = float(
        results.loc[
            results["treatment"]
            == observed_treatment,
            "p_label_1",
        ].iloc[0]
    )

    return (
        results,
        best_treatment,
        observed_treatment,
        best_probability,
        observed_probability,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-20")
    print("TREATMENT DECISION VALIDATION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "Parameters are learned from DEVELOPMENT "
        "data only."
    )
    print(
        "TEST data is used only for final evaluation."
    )
    print(
        "The final 23-edge DAG is NOT modified."
    )

    # ========================================================
    # DEVELOPMENT DATA
    # ========================================================

    print()
    print(
        "Loading development dataset..."
    )

    development = load_dataset(
        DEVELOPMENT_PATH,
        "Development",
    )

    print(
        f"Development shape: "
        f"{development.shape}"
    )

    print(
        f"Development missing: "
        f"{int(development.isna().sum().sum())}"
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    print()
    print(
        "Loading test dataset..."
    )

    test = load_dataset(
        TEST_PATH,
        "Test",
    )

    print(
        f"Test shape: "
        f"{test.shape}"
    )

    print(
        f"Test missing: "
        f"{int(test.isna().sum().sum())}"
    )

    # ========================================================
    # DAG
    # ========================================================

    print()
    print(
        "Loading final 23-edge DAG..."
    )

    edges = load_edges()

    print(
        f"Final DAG edges: "
        f"{len(edges)}"
    )

    # ========================================================
    # TEST STATE VALIDATION
    # ========================================================

    print()
    print(
        "Checking test states against "
        "development states..."
    )

    validate_test_states(
        development,
        test,
    )

    print(
        "All test states represented: PASSED"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()
    print(
        "Learning Bayesian Network parameters..."
    )

    model = build_model(
        development,
        edges,
    )

    print(
        "DAG structure: PASSED"
    )

    print(
        "Forbidden label -> trt: PASSED"
    )

    print(
        "Treatment -> outcome: PASSED"
    )

    print(
        "Model consistency: PASSED"
    )

    print(
        f"CPDs learned: "
        f"{len(model.get_cpds())}"
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    print()
    print(
        "Evaluating treatment decisions "
        "on TEST patients..."
    )

    total_patients = len(test)

    detailed_rows = []

    selected_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    observed_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    recommended_better_count = 0
    same_as_observed_count = 0

    patient_decisions = []

    for patient_number, (_, patient) in enumerate(
        test.iterrows(),
        start=1,
    ):

        (
            treatment_results,
            best_treatment,
            observed_treatment,
            best_probability,
            observed_probability,
        ) = evaluate_patient(
            model,
            patient,
        )

        # Count recommended treatment.
        selected_counts[
            best_treatment
        ] += 1

        # Count observed treatment.
        observed_counts[
            observed_treatment
        ] += 1

        # Compare predicted risk.
        if (
            best_probability
            < observed_probability
        ):

            recommended_better_count += 1

        # Compare recommendation with observed treatment.
        if (
            best_treatment
            == observed_treatment
        ):

            same_as_observed_count += 1

        patient_decisions.append(
            {
                "test_patient":
                    patient_number,

                "observed_treatment":
                    observed_treatment,

                "recommended_treatment":
                    best_treatment,

                "recommended_p_label_1":
                    best_probability,

                "observed_treatment_p_label_1":
                    observed_probability,

                "recommended_has_lower_predicted_risk":
                    int(
                        best_probability
                        < observed_probability
                    ),

                "recommendation_matches_observed":
                    int(
                        best_treatment
                        == observed_treatment
                    ),
            }
        )

        # Save predictions for all four treatments.
        for _, treatment_row in (
            treatment_results.iterrows()
        ):

            treatment_number = int(
                treatment_row[
                    "treatment"
                ]
            )

            detailed_rows.append(
                {
                    "test_patient":
                        patient_number,

                    "observed_treatment":
                        observed_treatment,

                    "recommended_treatment":
                        best_treatment,

                    "treatment":
                        treatment_number,

                    "treatment_name":
                        treatment_row[
                            "treatment_name"
                        ],

                    "short_name":
                        treatment_row[
                            "short_name"
                        ],

                    "p_label_0":
                        float(
                            treatment_row[
                                "p_label_0"
                            ]
                        ),

                    "p_label_1":
                        float(
                            treatment_row[
                                "p_label_1"
                            ]
                        ),

                    "recommended":
                        int(
                            best_treatment
                            == treatment_number
                        ),

                    "observed_treatment_row":
                        int(
                            observed_treatment
                            == treatment_number
                        ),
                }
            )

        if (
            patient_number % 25 == 0
            or patient_number
            == total_patients
        ):

            print(
                f"Processed "
                f"{patient_number}/"
                f"{total_patients}"
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    detailed_results = pd.DataFrame(
        detailed_rows
    )

    patient_decisions_df = pd.DataFrame(
        patient_decisions
    )

    # ========================================================
    # RECOMMENDATION DISTRIBUTION
    # ========================================================

    recommendation_rows = []

    for treatment in [
        0,
        1,
        2,
        3,
    ]:

        recommendation_rows.append(
            {
                "treatment":
                    treatment,

                "treatment_name":
                    TREATMENTS[
                        str(treatment)
                    ]["name"],

                "recommended_count":
                    selected_counts[
                        treatment
                    ],

                "recommended_rate":
                    selected_counts[
                        treatment
                    ]
                    / total_patients,

                "observed_count":
                    observed_counts[
                        treatment
                    ],

                "observed_rate":
                    observed_counts[
                        treatment
                    ]
                    / total_patients,
            }
        )

    recommendation_summary = (
        pd.DataFrame(
            recommendation_rows
        )
    )

    # ========================================================
    # TREATMENT-LEVEL SUMMARY
    # ========================================================

    treatment_summary = (
        detailed_results
        .groupby(
            [
                "treatment",
                "treatment_name",
            ],
            as_index=False,
        )
        .agg(
            mean_p_label_0=(
                "p_label_0",
                "mean",
            ),
            mean_p_label_1=(
                "p_label_1",
                "mean",
            ),
            median_p_label_1=(
                "p_label_1",
                "median",
            ),
            std_p_label_1=(
                "p_label_1",
                "std",
            ),
            min_p_label_1=(
                "p_label_1",
                "min",
            ),
            max_p_label_1=(
                "p_label_1",
                "max",
            ),
        )
    )

    # ========================================================
    # BEST AVERAGE TREATMENT
    # ========================================================

    best_average_index = (
        treatment_summary[
            "mean_p_label_1"
        ].idxmin()
    )

    best_average_treatment = int(
        treatment_summary.loc[
            best_average_index,
            "treatment",
        ]
    )

    best_average_treatment_name = (
        TREATMENTS[
            str(best_average_treatment)
        ]["name"]
    )

    # ========================================================
    # PATIENT TREATMENT RANKINGS
    # ========================================================

    ranking_rows = []

    for patient_number in range(
        1,
        total_patients + 1,
    ):

        patient_results = (
            detailed_results[
                detailed_results[
                    "test_patient"
                ]
                == patient_number
            ]
            .sort_values(
                "p_label_1"
            )
            .reset_index(
                drop=True
            )
        )

        if len(patient_results) != 4:
            continue

        ranking_rows.append(
            {
                "test_patient":
                    patient_number,

                "rank_1":
                    int(
                        patient_results.loc[
                            0,
                            "treatment"
                        ]
                    ),

                "rank_2":
                    int(
                        patient_results.loc[
                            1,
                            "treatment"
                        ]
                    ),

                "rank_3":
                    int(
                        patient_results.loc[
                            2,
                            "treatment"
                        ]
                    ),

                "rank_4":
                    int(
                        patient_results.loc[
                            3,
                            "treatment"
                        ]
                    ),
            }
        )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    recommended_better_rate = (
        recommended_better_count
        / total_patients
    )

    same_as_observed_rate = (
        same_as_observed_count
        / total_patients
    )

    # ========================================================
    # SAVE OUTPUT FILES
    # ========================================================

    detailed_path = (
        OUTPUT_DIR
        / "patient_treatment_predictions.csv"
    )

    treatment_summary_path = (
        OUTPUT_DIR
        / "treatment_summary.csv"
    )

    recommendation_path = (
        OUTPUT_DIR
        / "recommendation_distribution.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "treatment_rankings.csv"
    )

    patient_decisions_path = (
        OUTPUT_DIR
        / "patient_decisions.csv"
    )

    metrics_path = (
        OUTPUT_DIR
        / "decision_validation_metrics.json"
    )

    detailed_results.to_csv(
        detailed_path,
        index=False,
    )

    treatment_summary.to_csv(
        treatment_summary_path,
        index=False,
    )

    recommendation_summary.to_csv(
        recommendation_path,
        index=False,
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    patient_decisions_df.to_csv(
        patient_decisions_path,
        index=False,
    )

    metrics = {

        "phase": 20,

        "validation":
            "Treatment decision validation",

        "development_shape":
            list(
                development.shape
            ),

        "test_shape":
            list(
                test.shape
            ),

        "parameter_learning":
            "development_only",

        "test_used_for_parameter_learning":
            False,

        "final_dag_edges":
            len(edges),

        "test_patients":
            total_patients,

        "treatments_evaluated":
            4,

        "recommended_better_than_observed_rate":
            recommended_better_rate,

        "recommended_same_as_observed_rate":
            same_as_observed_rate,

        "best_average_treatment":
            best_average_treatment,

        "best_average_treatment_name":
            best_average_treatment_name,

        "recommendation_counts":
            selected_counts,

        "observed_treatment_counts":
            observed_counts,

        "interpretation":
            (
                "The observed treatment is not "
                "treated as the true optimal "
                "treatment. Validation compares "
                "model-based interventional "
                "rankings with the observed "
                "treatment and observed outcomes "
                "without claiming individual-level "
                "causal ground truth."
            ),
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE-20 RESULTS")
    print("=" * 70)

    print()
    print(
        "Treatment recommendation distribution:"
    )

    print(
        recommendation_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Average interventional predictions:"
    )

    print(
        treatment_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Best average treatment:"
    )

    print(
        f"Treatment "
        f"{best_average_treatment} — "
        f"{best_average_treatment_name}"
    )

    print()
    print(
        "Recommended treatment had a "
        "lower model-predicted P(label=1) "
        "than the observed treatment for:"
    )

    print(
        f"{recommended_better_count}/"
        f"{total_patients} "
        f"({recommended_better_rate:.2%})"
    )

    print()
    print(
        "Model recommendation matched "
        "the observed treatment for:"
    )

    print(
        f"{same_as_observed_count}/"
        f"{total_patients} "
        f"({same_as_observed_rate:.2%})"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These statistics do NOT prove that "
        "the recommended treatment is the "
        "true optimal treatment for each patient."
    )

    print(
        "ACTG175 only provides the observed "
        "treatment and observed outcome."
    )

    print(
        "This phase evaluates the stability "
        "and consistency of the model-based "
        "decision strategy."
    )

    # ========================================================
    # SAVED FILES
    # ========================================================

    print()
    print(
        "Saved detailed predictions:"
    )

    print(
        detailed_path
    )

    print()
    print(
        "Saved treatment summary:"
    )

    print(
        treatment_summary_path
    )

    print()
    print(
        "Saved recommendation distribution:"
    )

    print(
        recommendation_path
    )

    print()
    print(
        "Saved treatment rankings:"
    )

    print(
        ranking_path
    )

    print()
    print(
        "Saved patient decisions:"
    )

    print(
        patient_decisions_path
    )

    print()
    print(
        "Saved validation metrics:"
    )

    print(
        metrics_path
    )

    print()
    print("=" * 70)
    print("PHASE-20 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()