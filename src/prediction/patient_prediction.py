from pathlib import Path
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
# LOAD DATA
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
# LOAD FINAL DAG
# ============================================================

def load_edges():

    if not DAG_PATH.exists():
        raise FileNotFoundError(
            f"Final DAG not found:\n{DAG_PATH}"
        )

    edges_df = pd.read_csv(DAG_PATH)

    edges = []

    for _, row in edges_df.iterrows():

        source = str(row["source"]).strip()
        target = str(row["target"]).strip()

        edges.append((source, target))

    return edges


# ============================================================
# BUILD FINAL BAYESIAN NETWORK
# ============================================================

def build_model(data, edges):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(VARIABLES)
    model.add_edges_from(edges)

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
            "Final Bayesian Network failed consistency check."
        )

    return model


# ============================================================
# GET VALID STATES
# ============================================================

def get_states(data):

    states = {}

    for column in VARIABLES:

        states[column] = sorted(
            data[column].unique().tolist()
        )

    return states


# ============================================================
# INPUT PATIENT DATA
# ============================================================

def get_patient_input(data):

    states = get_states(data)

    print()
    print("=" * 70)
    print("PATIENT INPUT")
    print("=" * 70)

    print(
        "\nEnter the patient's values."
    )

    evidence = {}

    input_variables = [
        variable
        for variable in VARIABLES
        if variable != TARGET
    ]

    for variable in input_variables:

        valid_states = states[variable]

        print()
        print(
            f"{variable}"
        )

        print(
            f"Available values: "
            f"{', '.join(valid_states)}"
        )

        while True:

            value = input(
                "Enter value: "
            ).strip()

            if value in valid_states:

                evidence[variable] = value
                break

            print(
                "Invalid value."
            )

            print(
                "Please choose one of:"
                f" {', '.join(valid_states)}"
            )

    return evidence


# ============================================================
# PREDICTION
# ============================================================

def predict(model, evidence):

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
            probability_0 = float(probability)

        elif str(state) == "1":
            probability_1 = float(probability)

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

    if predicted_outcome == "1":

        print(
            "Predicted outcome: LABEL 1"
        )

    else:

        print(
            "Predicted outcome: LABEL 0"
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

    print()
    print(
        "Loading development data..."
    )

    data = load_data()

    print(
        f"Dataset shape: {data.shape}"
    )

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

    evidence = get_patient_input(
        data
    )

    probability_0, probability_1, predicted_outcome = predict(
        model,
        evidence,
    )

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


if __name__ == "__main__":
    main()