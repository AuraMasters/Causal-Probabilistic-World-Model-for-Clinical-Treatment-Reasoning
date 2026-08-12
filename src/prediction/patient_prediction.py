from pathlib import Path
import json
import warnings

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
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

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "discretization_metadata.json"
)


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

ESS = 10


# ============================================================
# NUMERICAL VARIABLES
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

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    for column in VARIABLES:
        data[column] = data[column].astype(str)

    return data


# ============================================================
# LOAD DISCRETIZATION METADATA
# ============================================================

def load_discretization_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Discretization metadata not found:\n"
            f"{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if metadata.get("representation") != "sparse":
        raise ValueError(
            "Expected sparse discretization metadata."
        )

    if metadata.get("fit_dataset") != "development_only":
        raise ValueError(
            "Discretization metadata was not fitted "
            "on development data only."
        )

    return metadata


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
            "Final DAG file must contain "
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
# BUILD FINAL BAYESIAN NETWORK
# ============================================================

def build_model(
    data,
    edges,
):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        VARIABLES
    )

    model.add_edges_from(
        edges
    )

    estimator = BayesianEstimator(
        model,
        data,
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
            "Final Bayesian Network "
            "failed consistency check."
        )

    return model


# ============================================================
# GET CATEGORICAL STATES
# ============================================================

def get_states(data):

    states = {}

    for column in VARIABLES:

        states[column] = sorted(
            data[column]
            .unique()
            .tolist()
        )

    return states


# ============================================================
# NUMERICAL → MODEL STATE
# ============================================================

def convert_numerical_value(
    variable,
    value,
    metadata,
):

    value = float(value)

    if variable not in metadata["variables"]:

        raise ValueError(
            f"No discretization metadata found "
            f"for {variable}."
        )

    variable_metadata = (
        metadata["variables"][variable]
    )

    edges = variable_metadata["edges"]

    # --------------------------------------------------------
    # preanti
    # --------------------------------------------------------

    if variable == "preanti":

        if value < 0:

            raise ValueError(
                "Pre-ART exposure cannot be negative."
            )

        if value == 0:

            return "zero"

        # Positive bins use:
        # 0
        # 366
        # 848.333333...
        # infinity

        if value <= edges[1]:

            return "positive_1"

        elif value <= edges[2]:

            return "positive_2"

        else:

            return "positive_3"

    # --------------------------------------------------------
    # Ordinary quantile variables
    # --------------------------------------------------------

    if variable in [
        "age",
        "wtkg",
        "cd40",
        "cd80",
    ]:

        if value <= edges[1]:

            return f"{variable}_1"

        elif value <= edges[2]:

            return f"{variable}_2"

        else:

            return f"{variable}_3"

    # --------------------------------------------------------
    # Karnofsky
    # --------------------------------------------------------

    if variable == "karnof":

        if value <= edges[1]:

            return "karnof_1"

        else:

            return "karnof_2"

    raise ValueError(
        f"Unsupported numerical variable: "
        f"{variable}"
    )


# ============================================================
# DISPLAY NUMERICAL RANGES
# ============================================================

def get_range_text(
    variable,
    metadata,
):

    edges = (
        metadata["variables"][variable]["edges"]
    )

    # --------------------------------------------------------
    # Karnofsky
    # --------------------------------------------------------

    if variable == "karnof":

        return (
            f"<= {edges[1]} -> karnof_1\n"
            f"> {edges[1]} -> karnof_2"
        )

    # --------------------------------------------------------
    # preanti
    # --------------------------------------------------------

    if variable == "preanti":

        return (
            "0 -> zero\n"
            f"0 < value <= {edges[1]} "
            "-> positive_1\n"
            f"{edges[1]} < value <= {edges[2]} "
            "-> positive_2\n"
            f"> {edges[2]} -> positive_3"
        )

    # --------------------------------------------------------
    # Three-bin variables
    # --------------------------------------------------------

    return (
        f"<= {edges[1]} -> {variable}_1\n"
        f"{edges[1]} < value <= {edges[2]} "
        f"-> {variable}_2\n"
        f"> {edges[2]} -> {variable}_3"
    )


# ============================================================
# GET NUMERICAL INPUT
# ============================================================

def get_numeric_input(
    variable,
    metadata,
):

    print()
    print(
        f"{variable}"
    )

    print(
        "Input ranges used by the model:"
    )

    print(
        get_range_text(
            variable,
            metadata,
        )
    )

    while True:

        value = input(
            "Enter numerical value: "
        ).strip()

        try:

            value = float(value)

            state = convert_numerical_value(
                variable,
                value,
                metadata,
            )

            return value, state

        except ValueError as error:

            print(
                f"Invalid value: {error}"
            )


# ============================================================
# GET BINARY INPUT
# ============================================================

def get_binary_input(
    variable,
    states,
):

    valid_states = [
        state
        for state in states[variable]
        if state in ["0", "1"]
    ]

    while True:

        value = input(
            f"{variable} "
            f"[{', '.join(valid_states)}]: "
        ).strip()

        if value in valid_states:

            return value

        print(
            "Invalid value."
        )


# ============================================================
# GET CHOICE INPUT
# ============================================================

def get_choice_input(
    variable,
    states,
):

    valid_states = states[variable]

    while True:

        value = input(
            f"{variable} "
            f"[{', '.join(valid_states)}]: "
        ).strip()

        if value in valid_states:

            return value

        print(
            "Invalid value."
        )


# ============================================================
# PATIENT INPUT
# ============================================================

def get_patient_input(
    data,
    metadata,
):

    states = get_states(
        data
    )

    print()
    print("=" * 70)
    print("PATIENT INPUT")
    print("=" * 70)

    print()
    print(
        "Enter real numerical values for "
        "the discretized variables."
    )

    evidence = {}

    numerical_inputs = {}

    # --------------------------------------------------------
    # Numerical variables
    # --------------------------------------------------------

    for variable in NUMERICAL_VARIABLES:

        value, state = (
            get_numeric_input(
                variable,
                metadata,
            )
        )

        numerical_inputs[
            variable
        ] = value

        evidence[
            variable
        ] = state

    # --------------------------------------------------------
    # Binary variables
    # --------------------------------------------------------

    binary_variables = [
        "hemo",
        "homo",
        "drugs",
        "oprior",
        "z30",
        "race",
        "gender",
        "symptom",
    ]

    print()

    for variable in binary_variables:

        evidence[
            variable
        ] = get_binary_input(
            variable,
            states,
        )

    # --------------------------------------------------------
    # Stratification
    # --------------------------------------------------------

    print()

    evidence["strat"] = (
        get_choice_input(
            "strat",
            states,
        )
    )

    # --------------------------------------------------------
    # Treatment
    # --------------------------------------------------------

    print()

    evidence["trt"] = (
        get_choice_input(
            "trt",
            states,
        )
    )

    return (
        evidence,
        numerical_inputs,
    )


# ============================================================
# DISPLAY CONVERSION
# ============================================================

def display_conversion(
    numerical_inputs,
    evidence,
):

    print()
    print("=" * 70)
    print("NUMERICAL INPUT → BAYESIAN NETWORK STATE")
    print("=" * 70)

    for variable in NUMERICAL_VARIABLES:

        print(
            f"{variable:<10}: "
            f"{numerical_inputs[variable]} "
            f"-> "
            f"{evidence[variable]}"
        )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    evidence,
):

    inference = VariableElimination(
        model
    )

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

    predicted_outcome = (
        "1"
        if probability_1 >= probability_0
        else "0"
    )

    return (
        probability_0,
        probability_1,
        predicted_outcome,
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    evidence,
    probability_0,
    probability_1,
    predicted_outcome,
):

    print()
    print("=" * 70)
    print("PATIENT PREDICTION RESULT")
    print("=" * 70)

    print()
    print("Patient evidence:")

    for variable, value in evidence.items():

        print(
            f"  {variable}: {value}"
        )

    print()

    print(
        f"P(label = 0): "
        f"{probability_0:.6f}"
    )

    print(
        f"P(label = 1): "
        f"{probability_1:.6f}"
    )

    print()

    print(
        f"Predicted outcome: "
        f"LABEL {predicted_outcome}"
    )

    print()

    print(
        f"Prediction confidence: "
        f"{max(probability_0, probability_1) * 100:.2f}%"
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-17")
    print("PATIENT PROBABILISTIC PREDICTION")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Load discretization metadata
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
        f"{metadata['fit_dataset']}"
    )

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    print()
    print(
        "Loading final 23-edge DAG..."
    )

    edges = load_edges()

    print(
        f"Final DAG edges: {len(edges)}"
    )

    if len(edges) != 23:

        raise ValueError(
            f"Expected 23 edges, "
            f"found {len(edges)}"
        )

    # --------------------------------------------------------
    # Build Bayesian Network
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

    print()
    print(
        "Creating inference engine..."
    )

    # --------------------------------------------------------
    # Patient input
    # --------------------------------------------------------

    (
        evidence,
        numerical_inputs,
    ) = get_patient_input(
        data,
        metadata,
    )

    # --------------------------------------------------------
    # Show numerical conversion
    # --------------------------------------------------------

    display_conversion(
        numerical_inputs,
        evidence,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    (
        probability_0,
        probability_1,
        predicted_outcome,
    ) = predict(
        model,
        evidence,
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    display_result(
        evidence,
        probability_0,
        probability_1,
        predicted_outcome,
    )

    print()
    print(
        "PHASE-17 COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()