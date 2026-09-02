import json
from pathlib import Path

import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# ============================================================
# ACTG175 PHASE-10 PROBABILISTIC INFERENCE ENGINE
# pgmpy 1.1.2 compatible
# ============================================================


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
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

    df = pd.read_csv(DATA_PATH)

    required = set(VARIABLES)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing variables:\n"
            + "\n".join(sorted(missing))
        )

    df = df[VARIABLES].copy()

    print(f"Development dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")

    return df


# ============================================================
# LOAD DAG
# ============================================================

def load_dag():

    if not DAG_PATH.exists():
        raise FileNotFoundError(
            f"Final DAG not found:\n{DAG_PATH}"
        )

    edges_df = pd.read_csv(DAG_PATH)

    if "source" not in edges_df.columns:
        raise ValueError(
            "DAG file does not contain 'source'."
        )

    if "target" not in edges_df.columns:
        raise ValueError(
            "DAG file does not contain 'target'."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(row["source"]).strip()
        target = str(row["target"]).strip()

        edges.append(
            (source, target)
        )

    edges = list(dict.fromkeys(edges))

    print(f"\nFinal DAG: {DAG_PATH}")
    print(f"Edges: {len(edges)}")

    return edges


# ============================================================
# BUILD NETWORK
# ============================================================

def build_network(
    df,
    edges,
):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(VARIABLES)
    model.add_edges_from(edges)

    print("\nFinal DAG edges:")

    for source, target in sorted(edges):
        print(
            f"{source} -> {target}"
        )

    return model


# ============================================================
# LEARN CPDS
# ============================================================

def learn_cpds(
    model,
    df,
):

    print("\nLearning BDeu parameters...")

    estimator = BayesianEstimator(
        model,
        df,
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
            "Bayesian network consistency check failed."
        )

    print(
        f"CPDs learned: {len(cpds)}"
    )

    print(
        "Model consistency: PASSED"
    )

    return model


# ============================================================
# CREATE INFERENCE OBJECT
# ============================================================

def create_inference_engine(
    model,
):

    print(
        "\nCreating Variable Elimination engine..."
    )

    inference = VariableElimination(
        model
    )

    print(
        "Inference engine: READY"
    )

    return inference


# ============================================================
# BASIC QUERY
# ============================================================

def query_probability(
    inference,
    variable,
    evidence=None,
):

    if evidence is None:
        evidence = {}

    result = inference.query(
        variables=[variable],
        evidence=evidence,
        show_progress=False,
    )

    return result


# ============================================================
# PRINT QUERY
# ============================================================

def print_query(
    inference,
    variable,
    evidence=None,
):

    if evidence is None:
        evidence = {}

    result = query_probability(
        inference,
        variable,
        evidence,
    )

    print("\n" + "-" * 70)

    print(
        f"QUERY: P({variable} | evidence)"
    )

    if evidence:
        print(
            f"Evidence: {evidence}"
        )
    else:
        print(
            "Evidence: None"
        )

    print("-" * 70)

    print(result)

    # --------------------------------------------------------
    # Print individual states
    # --------------------------------------------------------

    if hasattr(
        result,
        "state_names",
    ):

        states = result.state_names.get(
            variable,
            [],
        )

        values = result.values

        for state, probability in zip(
            states,
            values,
        ):

            print(
                f"P({variable}={state}) = "
                f"{float(probability):.6f}"
            )

    return result


# ============================================================
# SAVE QUERY RESULT
# ============================================================

def result_to_dict(
    result,
    variable,
    evidence,
):

    states = result.state_names.get(
        variable,
        [],
    )

    probabilities = result.values

    rows = []

    for state, probability in zip(
        states,
        probabilities,
    ):

        rows.append(
            {
                "query_variable": variable,
                "state": str(state),
                "probability":
                    float(probability),
                "evidence":
                    json.dumps(
                        evidence,
                        sort_keys=True,
                    ),
            }
        )

    return rows


# ============================================================
# RUN STANDARD QUERIES
# ============================================================

def run_standard_queries(
    inference,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STANDARD PROBABILISTIC QUERIES"
    )

    print(
        "=" * 70
    )

    all_results = []

    # --------------------------------------------------------
    # 1. Marginal outcome probability
    # --------------------------------------------------------

    result = print_query(
        inference,
        "label",
    )

    all_results.extend(
        result_to_dict(
            result,
            "label",
            {},
        )
    )

    # --------------------------------------------------------
    # 2. Treatment-specific outcome queries
    # --------------------------------------------------------

    for treatment in [0, 1, 2, 3]:

        result = print_query(
            inference,
            "label",
            {
                "trt": treatment
            },
        )

        all_results.extend(
            result_to_dict(
                result,
                "label",
                {
                    "trt": treatment
                },
            )
        )

    # --------------------------------------------------------
    # 3. z30-specific outcome
    # --------------------------------------------------------

    for state in [0, 1]:

        result = print_query(
            inference,
            "label",
            {
                "z30": state
            },
        )

        all_results.extend(
            result_to_dict(
                result,
                "label",
                {
                    "z30": state
                },
            )
        )

    # --------------------------------------------------------
    # 4. CD40-specific outcome
    # --------------------------------------------------------

    cd40_states = [
        "cd40_1",
        "cd40_2",
        "cd40_3",
    ]

    for state in cd40_states:

        result = print_query(
            inference,
            "label",
            {
                "cd40": state
            },
        )

        all_results.extend(
            result_to_dict(
                result,
                "label",
                {
                    "cd40": state
                },
            )
        )

    return all_results


# ============================================================
# MULTI-EVIDENCE QUERY
# ============================================================

def run_multi_evidence_query(
    inference,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-EVIDENCE INFERENCE"
    )

    print(
        "=" * 70
    )

    evidence = {
        "trt": 1,
        "age": "age_2",
        "homo": 1,
        "symptom": 0,
    }

    print(
        "\nExample patient evidence:"
    )

    for key, value in evidence.items():

        print(
            f"  {key} = {value}"
        )

    result = print_query(
        inference,
        "label",
        evidence,
    )

    return result


# ============================================================
# TREATMENT COMPARISON
# ============================================================

def treatment_comparison(
    inference,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TREATMENT OUTCOME COMPARISON"
    )

    print(
        "=" * 70
    )

    rows = []

    for treatment in [0, 1, 2, 3]:

        result = query_probability(
            inference,
            "label",
            {
                "trt": treatment
            },
        )

        states = result.state_names[
            "label"
        ]

        probabilities = result.values

        probability_positive = None

        for state, probability in zip(
            states,
            probabilities,
        ):

            if str(state) == "1":

                probability_positive = float(
                    probability
                )

        if probability_positive is None:

            raise ValueError(
                "label=1 state not found."
            )

        rows.append(
            {
                "treatment": treatment,
                "P_label_1":
                    probability_positive,
                "P_label_0":
                    1.0
                    - probability_positive,
            }
        )

    comparison_df = pd.DataFrame(
        rows
    )

    print(
        "\n"
        + comparison_df.to_string(
            index=False
        )
    )

    return comparison_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    rows,
):

    output_path = (
        OUTPUT_DIR
        / "standard_inference_results.csv"
    )

    results_df = pd.DataFrame(
        rows
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved inference results:"
        f"\n{output_path}"
    )

    return results_df


# ============================================================
# SAVE TREATMENT COMPARISON
# ============================================================

def save_treatment_comparison(
    comparison_df,
):

    output_path = (
        OUTPUT_DIR
        / "treatment_comparison.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved treatment comparison:"
        f"\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ACTG175 PHASE-10 PROBABILISTIC INFERENCE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    edges = load_dag()

    # --------------------------------------------------------
    # Build BN
    # --------------------------------------------------------

    model = build_network(
        df,
        edges,
    )

    # --------------------------------------------------------
    # Learn parameters
    # --------------------------------------------------------

    model = learn_cpds(
        model,
        df,
    )

    # --------------------------------------------------------
    # Inference engine
    # --------------------------------------------------------

    inference = create_inference_engine(
        model
    )

    # --------------------------------------------------------
    # Standard queries
    # --------------------------------------------------------

    results = run_standard_queries(
        inference
    )

    # --------------------------------------------------------
    # Multi-evidence query
    # --------------------------------------------------------

    run_multi_evidence_query(
        inference
    )

    # --------------------------------------------------------
    # Treatment comparison
    # --------------------------------------------------------

    comparison = treatment_comparison(
        inference
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    save_treatment_comparison(
        comparison
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "phase":
            "Phase 10 - Probabilistic Inference",

        "dataset":
            "ACTG175",

        "representation":
            "sparse",

        "rows":
            len(df),

        "variables":
            len(VARIABLES),

        "dag_edges":
            len(edges),

        "inference_algorithm":
            "Variable Elimination",

        "parameter_estimator":
            "BayesianEstimator",

        "prior":
            "BDeu",

        "equivalent_sample_size":
            ESS,

        "queries_executed":
            len(results),

        "model_check":
            True,
    }

    metadata_path = (
        OUTPUT_DIR
        / "inference_metadata.json"
    )

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

    print(
        f"Saved metadata:"
        f"\n{metadata_path}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE-10 INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()