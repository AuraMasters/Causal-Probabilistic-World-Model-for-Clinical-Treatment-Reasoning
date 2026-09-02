import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# PHASE 18
# INTERVENTIONAL TREATMENT ANALYSIS
# ============================================================

PHASE_NAME = "ACTG175 PHASE-18"
ESS = 10


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_model"
    / "dag"
    / "final_dag_edges.csv"
)

DISCRETIZATION_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "discretization_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
    / "intervention"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VARIABLES
# ============================================================

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
# DISCRETIZED VARIABLES
# ============================================================

NUMERICAL_VARIABLES = [
    "age",
    "wtkg",
    "karnof",
    "preanti",
    "cd40",
    "cd80",
]


# ============================================================
# LOAD DEVELOPMENT DATA
# ============================================================

def load_data():
    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n{DEVELOPMENT_PATH}"
        )

    data = pd.read_csv(DEVELOPMENT_PATH)

    missing_columns = [
        column
        for column in VARIABLES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Development dataset is missing columns:\n"
            + ", ".join(missing_columns)
        )

    for column in VARIABLES:
        data[column] = data[column].astype(str)

    return data


# ============================================================
# LOAD DISCRETIZATION METADATA
# ============================================================

def load_discretization_metadata():
    if not DISCRETIZATION_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Discretization metadata not found:\n"
            f"{DISCRETIZATION_METADATA_PATH}"
        )

    with open(
        DISCRETIZATION_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return metadata


# ============================================================
# LOAD DAG
# ============================================================

def load_edges():
    if not DAG_PATH.exists():
        raise FileNotFoundError(
            f"Final DAG not found:\n{DAG_PATH}"
        )

    edges_df = pd.read_csv(DAG_PATH)

    required_columns = {"source", "target"}

    if not required_columns.issubset(
        set(edges_df.columns)
    ):
        raise ValueError(
            "DAG file must contain "
            "'source' and 'target' columns."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(row["source"]).strip()
        target = str(row["target"]).strip()

        edges.append(
            (source, target)
        )

    return edges


# ============================================================
# VALIDATE DAG
# ============================================================

def validate_dag(model, edges):

    if len(edges) != 23:
        raise ValueError(
            f"Expected 23 final DAG edges, "
            f"found {len(edges)}."
        )

    if not nx_is_dag(model):
        raise ValueError(
            "Final DAG is not acyclic."
        )

    if ("label", "trt") in edges:
        raise ValueError(
            "Forbidden edge label -> trt detected."
        )

    if ("trt", "label") not in edges:
        raise ValueError(
            "Final DAG must contain trt -> label."
        )

    return True


def nx_is_dag(model):
    """
    Compatibility helper.

    pgmpy DiscreteBayesianNetwork inherits from
    NetworkX-compatible DAG functionality.
    """

    try:
        import networkx as nx

        return nx.is_directed_acyclic_graph(
            model
        )

    except Exception:  # noqa: BLE001
        # Fallback topological-sort check.
        try:
            list(model.topological_sort())
            return True
        except Exception:  # noqa: BLE001
            return False


# ============================================================
# BUILD AND LEARN BAYESIAN NETWORK
# ============================================================

def build_model(data, edges):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(VARIABLES)
    model.add_edges_from(edges)

    validate_dag(
        model,
        edges,
    )

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
        raise ValueError(
            "Bayesian Network failed consistency check."
        )

    return model


# ============================================================
# GET STATES
# ============================================================

def get_states(data):

    states = {}

    for column in VARIABLES:

        states[column] = sorted(
            data[column]
            .astype(str)
            .unique()
            .tolist()
        )

    return states


# ============================================================
# NUMERICAL DISCRETIZATION
# ============================================================

def discretize_numerical_value(
    variable,
    value,
    metadata,
):
    """
    Convert a real numerical patient value into the
    exact discretized state used by the final model.

    IMPORTANT:
    These boundaries come directly from the
    development-only discretization metadata.
    """

    variable_metadata = metadata[
        "variables"
    ][variable]

    method = variable_metadata.get(
        "method"
    )

    edges = variable_metadata[
        "edges"
    ]

    value = float(value)

    # --------------------------------------------------------
    # Standard quantile discretization
    # --------------------------------------------------------

    if method == "quantile":

        if variable == "age":

            if value <= edges[1]:
                return "age_1"

            elif value <= edges[2]:
                return "age_2"

            return "age_3"

        if variable == "wtkg":

            if value <= edges[1]:
                return "wtkg_1"

            elif value <= edges[2]:
                return "wtkg_2"

            return "wtkg_3"

        if variable == "karnof":

            if value <= edges[1]:
                return "karnof_1"

            return "karnof_2"

        if variable == "cd40":

            if value <= edges[1]:
                return "cd40_1"

            elif value <= edges[2]:
                return "cd40_2"

            return "cd40_3"

        if variable == "cd80":

            if value <= edges[1]:
                return "cd80_1"

            elif value <= edges[2]:
                return "cd80_2"

            return "cd80_3"

    # --------------------------------------------------------
    # preanti special discretization
    # --------------------------------------------------------

    if method == "zero_plus_positive_quantiles":

        if value == 0:
            return "zero"

        elif value <= edges[1]:
            return "positive_1"

        elif value <= edges[2]:
            return "positive_2"

        return "positive_3"

    raise ValueError(
        f"Unsupported discretization method "
        f"for {variable}: {method}"
    )


# ============================================================
# ASK NUMERICAL INPUT
# ============================================================

def ask_numerical_input(
    variable,
    metadata,
):
    variable_metadata = metadata[
        "variables"
    ][variable]

    edges = variable_metadata[
        "edges"
    ]

    print()
    print(variable)

    print(
        "Input ranges used by the model:"
    )

    if variable == "age":

        print(
            f"<= {edges[1]} -> age_1"
        )

        print(
            f"{edges[1]} < value <= "
            f"{edges[2]} -> age_2"
        )

        print(
            f"> {edges[2]} -> age_3"
        )

    elif variable == "wtkg":

        print(
            f"<= {edges[1]} -> wtkg_1"
        )

        print(
            f"{edges[1]} < value <= "
            f"{edges[2]} -> wtkg_2"
        )

        print(
            f"> {edges[2]} -> wtkg_3"
        )

    elif variable == "karnof":

        print(
            f"<= {edges[1]} -> karnof_1"
        )

        print(
            f"> {edges[1]} -> karnof_2"
        )

    elif variable == "preanti":

        print(
            "0 -> zero"
        )

        print(
            f"0 < value <= "
            f"{edges[1]} -> positive_1"
        )

        print(
            f"{edges[1]} < value <= "
            f"{edges[2]} -> positive_2"
        )

        print(
            f"> {edges[2]} -> positive_3"
        )

    elif variable == "cd40":

        print(
            f"<= {edges[1]} -> cd40_1"
        )

        print(
            f"{edges[1]} < value <= "
            f"{edges[2]} -> cd40_2"
        )

        print(
            f"> {edges[2]} -> cd40_3"
        )

    elif variable == "cd80":

        print(
            f"<= {edges[1]} -> cd80_1"
        )

        print(
            f"{edges[1]} < value <= "
            f"{edges[2]} -> cd80_2"
        )

        print(
            f"> {edges[2]} -> cd80_3"
        )

    while True:

        raw = input(
            "Enter numerical value: "
        ).strip()

        try:

            value = float(raw)

            if not np.isfinite(value):
                raise ValueError

            state = discretize_numerical_value(
                variable,
                value,
                metadata,
            )

            print(
                f"{variable:<10}: "
                f"{value} -> {state}"
            )

            return state, value

        except ValueError:

            print(
                "Invalid numerical value. "
                "Please enter a valid number."
            )


# ============================================================
# ASK CATEGORICAL INPUT
# ============================================================

def ask_categorical_input(
    variable,
    states,
):

    valid_states = states[
        variable
    ]

    print()
    print(
        f"{variable} "
        f"[{', '.join(valid_states)}]"
    )

    while True:

        value = input(
            "Enter value: "
        ).strip()

        if value in valid_states:
            return value

        print(
            "Invalid value."
        )

        print(
            "Available values: "
            + ", ".join(valid_states)
        )


# ============================================================
# COLLECT PATIENT EVIDENCE
# ============================================================

def get_patient_evidence(
    data,
    metadata,
):

    states = get_states(data)

    print()
    print("=" * 70)
    print("PATIENT INPUT")
    print("=" * 70)

    print()
    print(
        "Enter the patient's clinical information."
    )

    print()
    print(
        "Numerical variables will be converted "
        "using the development-fitted bins."
    )

    evidence = {}

    original_numerical_values = {}

    # --------------------------------------------------------
    # Numerical variables
    # --------------------------------------------------------

    for variable in NUMERICAL_VARIABLES:

        state, original_value = (
            ask_numerical_input(
                variable,
                metadata,
            )
        )

        evidence[variable] = state

        original_numerical_values[
            variable
        ] = original_value

    # --------------------------------------------------------
    # Categorical variables
    # --------------------------------------------------------

    categorical_variables = [
        variable
        for variable in VARIABLES
        if variable not in NUMERICAL_VARIABLES
        and variable not in {
            TARGET,
            TREATMENT,
        }
    ]

    for variable in categorical_variables:

        evidence[variable] = (
            ask_categorical_input(
                variable,
                states,
            )
        )

    return (
        evidence,
        original_numerical_values,
    )


# ============================================================
# INTERVENTIONAL PREDICTION
# ============================================================

def intervention_probability(
    model,
    patient_evidence,
    treatment_value,
):
    """
    Calculate:

        P(label = 1 | patient evidence, do(trt=t))

    using the final Bayesian Network.

    The intervention is explicitly represented using
    pgmpy's do() operation.
    """

    intervention_model = model.do(
        [TREATMENT],
        inplace=False,
    )

    inference = VariableElimination(
        intervention_model
    )

    evidence = dict(
        patient_evidence
    )

    # The treatment is not treated as ordinary
    # patient evidence; it is fixed by intervention.
    evidence[TREATMENT] = str(
        treatment_value
    )

    result = inference.query(
        variables=[TARGET],
        evidence=evidence,
        show_progress=False,
    )

    target_states = result.state_names[
        TARGET
    ]

    probabilities = result.values

    probability_label_0 = 0.0
    probability_label_1 = 0.0

    for state, probability in zip(
        target_states,
        probabilities,
    ):

        if str(state) == "0":

            probability_label_0 = float(
                probability
            )

        elif str(state) == "1":

            probability_label_1 = float(
                probability
            )

    return (
        probability_label_0,
        probability_label_1,
    )


# ============================================================
# RUN ALL FOUR INTERVENTIONS
# ============================================================

def evaluate_all_treatments(
    model,
    patient_evidence,
):

    results = []

    print()
    print("=" * 70)
    print("INTERVENTIONAL TREATMENT ANALYSIS")
    print("=" * 70)

    print()
    print(
        "Evaluating all four ACTG175 treatment interventions..."
    )

    for treatment_value in [
        "0",
        "1",
        "2",
        "3",
    ]:

        (
            probability_0,
            probability_1,
        ) = intervention_probability(
            model,
            patient_evidence,
            treatment_value,
        )

        treatment_name = TREATMENTS[
            treatment_value
        ]["name"]

        short_name = TREATMENTS[
            treatment_value
        ]["short_name"]

        results.append(
            {
                "treatment": int(
                    treatment_value
                ),
                "treatment_name": treatment_name,
                "short_name": short_name,
                "p_label_0": probability_0,
                "p_label_1": probability_1,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Because label=1 is the failure/event indicator,
    # lower P(label=1) is considered better.
    # --------------------------------------------------------

    best_index = (
        results_df[
            "p_label_1"
        ].idxmin()
    )

    results_df[
        "rank_by_p_label_1"
    ] = (
        results_df[
            "p_label_1"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    results_df[
        "is_best_treatment"
    ] = False

    results_df.loc[
        best_index,
        "is_best_treatment",
    ] = True

    return results_df


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results_df,
):

    print()
    print("=" * 70)
    print("TREATMENT COMPARISON")
    print("=" * 70)

    print()

    for _, row in results_df.iterrows():

        print(
            f"Treatment {int(row['treatment'])}"
        )

        print(
            f"  {row['treatment_name']}"
        )

        print(
            f"  P(label = 0): "
            f"{row['p_label_0']:.6f}"
        )

        print(
            f"  P(label = 1): "
            f"{row['p_label_1']:.6f}"
        )

        print()

    best_row = results_df.loc[
        results_df[
            "is_best_treatment"
        ]
    ].iloc[0]

    print("-" * 70)

    print(
        "BEST TREATMENT UNDER THE MODEL"
    )

    print()

    print(
        f"Treatment "
        f"{int(best_row['treatment'])}"
    )

    print(
        f"{best_row['treatment_name']}"
    )

    print()

    print(
        f"Predicted P(label = 1): "
        f"{best_row['p_label_1']:.2%}"
    )

    print(
        f"Predicted P(label = 0): "
        f"{best_row['p_label_0']:.2%}"
    )

    print()

    print(
        "Selection criterion:"
    )

    print(
        "Lowest predicted probability "
        "of label = 1."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a model-based treatment "
        "comparison, not a guaranteed clinical "
        "recommendation."
    )

    print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    patient_evidence,
    original_numerical_values,
):

    predictions_path = (
        OUTPUT_DIR
        / "intervention_results.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / "intervention_metadata.json"
    )

    results_df.to_csv(
        predictions_path,
        index=False,
    )

    best_row = results_df.loc[
        results_df[
            "is_best_treatment"
        ]
    ].iloc[0]

    metadata = {

        "phase": 18,

        "analysis": (
            "Interventional treatment comparison"
        ),

        "model": (
            "Final 23-edge ACTG175 Bayesian Network"
        ),

        "parameter_learning": {
            "dataset": "development_only",
            "prior": "BDeu",
            "equivalent_sample_size": ESS,
        },

        "intervention": (
            "do(trt = treatment)"
        ),

        "target": TARGET,

        "treatment_variable": TREATMENT,

        "treatment_definitions": TREATMENTS,

        "patient_numerical_inputs": (
            original_numerical_values
        ),

        "patient_model_evidence": (
            patient_evidence
        ),

        "best_treatment": {
            "trt": int(
                best_row["treatment"]
            ),
            "name": best_row[
                "treatment_name"
            ],
            "p_label_0": float(
                best_row["p_label_0"]
            ),
            "p_label_1": float(
                best_row["p_label_1"]
            ),
        },

        "selection_rule": (
            "Choose treatment with the "
            "lowest predicted P(label=1)."
        ),

        "warning": (
            "Model-based comparison only. "
            "Not a clinical prescription."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print(
        "Saved intervention results:"
    )

    print(
        predictions_path
    )

    print()

    print(
        "Saved intervention metadata:"
    )

    print(
        metadata_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(PHASE_NAME)
    print(
        "INTERVENTIONAL TREATMENT COMPARISON"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The existing final 23-edge DAG "
        "will NOT be modified."
    )

    print(
        "Parameters are learned from "
        "DEVELOPMENT data only."
    )

    print(
        "This phase compares all four ACTG175 "
        "treatment interventions."
    )

    # --------------------------------------------------------
    # Load development data
    # --------------------------------------------------------

    print()
    print(
        "Loading development data..."
    )

    data = load_data()

    print(
        f"Dataset shape: {data.shape}"
    )

    print(
        f"Missing values: "
        f"{int(data.isna().sum().sum())}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print()
    print(
        "Loading discretization metadata..."
    )

    metadata = (
        load_discretization_metadata()
    )

    print(
        "Discretization metadata: READY"
    )

    print(
        "Fitted on: "
        + str(
            metadata.get(
                "fit_dataset",
                "unknown",
            )
        )
    )

    # --------------------------------------------------------
    # Load DAG
    # --------------------------------------------------------

    print()
    print(
        "Loading final 23-edge DAG..."
    )

    edges = load_edges()

    print(
        f"Final DAG edges: {len(edges)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print(
        "Building final Bayesian Network..."
    )

    model = build_model(
        data,
        edges,
    )

    print(
        "Model consistency: PASSED"
    )

    print(
        f"CPDs learned: "
        f"{len(model.get_cpds())}"
    )

    # --------------------------------------------------------
    # Patient input
    # --------------------------------------------------------

    (
        patient_evidence,
        original_numerical_values,
    ) = get_patient_evidence(
        data,
        metadata,
    )

    # --------------------------------------------------------
    # Display model evidence
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PATIENT MODEL EVIDENCE")
    print("=" * 70)

    for variable, value in (
        patient_evidence.items()
    ):

        print(
            f"{variable:<10}: {value}"
        )

    # --------------------------------------------------------
    # Evaluate interventions
    # --------------------------------------------------------

    results_df = (
        evaluate_all_treatments(
            model,
            patient_evidence,
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        results_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results_df,
        patient_evidence,
        original_numerical_values,
    )

    print()
    print("=" * 70)
    print("PHASE-18 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()