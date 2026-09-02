import itertools
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

# ============================================================
# ACTG175 PHASE-9 PARAMETER LEARNING
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

FINAL_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parameter_learning"
)

CPT_DIR = OUTPUT_DIR / "cpts"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CPT_DIR.mkdir(
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

# BDeu equivalent sample size
ESS = 10


# ============================================================
# EXPECTED FINAL DAG FROM PHASE 8
# ============================================================

EXPECTED_EDGES = [
    ("cd40", "label"),
    ("cd40", "symptom"),
    ("cd80", "cd40"),
    ("gender", "wtkg"),
    ("hemo", "age"),
    ("hemo", "drugs"),
    ("hemo", "gender"),
    ("hemo", "homo"),
    ("homo", "cd80"),
    ("homo", "drugs"),
    ("homo", "gender"),
    ("homo", "symptom"),
    ("preanti", "race"),
    ("preanti", "strat"),
    ("preanti", "z30"),
    ("race", "homo"),
    ("strat", "hemo"),
    ("strat", "oprior"),
    ("symptom", "karnof"),
    ("z30", "age"),
    ("z30", "label"),
    ("z30", "oprior"),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDevelopment dataset not found:\n"
            f"{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Development dataset:\n"
        f"{DATA_PATH}"
    )

    print(
        f"Shape: {df.shape}"
    )

    missing_columns = [
        column
        for column in VARIABLES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    df = df[VARIABLES].copy()

    missing_values = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"Missing values: {missing_values}"
    )

    print(
        f"Duplicate rows: {duplicate_rows}"
    )

    if missing_values > 0:
        raise ValueError(
            "Dataset contains missing values."
        )

    return df


# ============================================================
# LOAD FINAL DAG
# ============================================================

def load_final_dag():

    if not FINAL_DAG_PATH.exists():
        raise FileNotFoundError(
            f"\nFinal DAG file not found:\n"
            f"{FINAL_DAG_PATH}"
        )

    edges_df = pd.read_csv(
        FINAL_DAG_PATH
    )

    print(
        f"\nFinal DAG file:\n"
        f"{FINAL_DAG_PATH}"
    )

    print(
        f"Columns: "
        f"{edges_df.columns.tolist()}"
    )

    if (
        "source" not in edges_df.columns
        or "target" not in edges_df.columns
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

        if source and target:
            edges.append(
                (
                    source,
                    target,
                )
            )

    edges = list(
        dict.fromkeys(edges)
    )

    return edges


# ============================================================
# VALIDATE DAG
# ============================================================

def validate_graph(
    edges,
    df,
):

    print(
        "\nValidating final DAG..."
    )

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        VARIABLES
    )

    model.add_edges_from(
        edges
    )

    # --------------------------------------------------------
    # DAG validation
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(
        model
    ):

        cycles = list(
            nx.simple_cycles(model)
        )

        raise ValueError(
            "Final graph is NOT a DAG.\n"
            f"Cycles: {cycles}"
        )

    print(
        "DAG structure: PASSED"
    )

    # --------------------------------------------------------
    # Dataset schema
    # --------------------------------------------------------

    missing_columns = (
        set(VARIABLES)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset missing variables:\n"
            + "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    # --------------------------------------------------------
    # Temporal structure
    #
    # Baseline -> Treatment -> Outcome
    # --------------------------------------------------------

    baseline = {
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
    }

    treatment = {
        "trt"
    }

    outcome = {
        "label"
    }

    tier = {}

    for variable in baseline:
        tier[variable] = 0

    for variable in treatment:
        tier[variable] = 1

    for variable in outcome:
        tier[variable] = 2

    violations = []

    for source, target in edges:

        if tier[source] > tier[target]:

            violations.append(
                (
                    source,
                    target,
                )
            )

    if violations:

        for source, target in violations:
            print(
                f"{source} -> {target}"
            )

        raise ValueError(
            "Temporal constraints failed."
        )

    print(
        "Temporal constraints: PASSED"
    )

    # --------------------------------------------------------
    # Forbidden edge
    # --------------------------------------------------------

    if (
        "label",
        "trt",
    ) in edges:

        raise ValueError(
            "Forbidden edge found: "
            "label -> trt"
        )

    print(
        "Forbidden label -> trt: PASSED"
    )

    # --------------------------------------------------------
    # Exact Phase-8 DAG validation
    # --------------------------------------------------------

    actual_edges = set(
        edges
    )

    expected_edges = set(
        EXPECTED_EDGES
    )

    if actual_edges != expected_edges:

        missing = (
            expected_edges
            - actual_edges
        )

        extra = (
            actual_edges
            - expected_edges
        )

        if missing:

            print(
                "\nMissing expected edges:"
            )

            for source, target in sorted(
                missing
            ):
                print(
                    f"{source} -> {target}"
                )

        if extra:

            print(
                "\nUnexpected edges:"
            )

            for source, target in sorted(
                extra
            ):
                print(
                    f"{source} -> {target}"
                )

        raise ValueError(
            "The loaded DAG does not match "
            "the frozen Phase-8 DAG."
        )

    print(
        "Final DAG identity: PASSED"
    )

    return model


# ============================================================
# PARAMETER LEARNING
# ============================================================

def learn_parameters(
    model,
    df,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "PARAMETER LEARNING"
    )

    print(
        "=" * 70
    )

    print(
        "Estimator: BayesianEstimator"
    )

    print(
        "Prior: BDeu"
    )

    print(
        f"Equivalent sample size: {ESS}"
    )

    print(
        "\nLearning CPTs..."
    )

    # --------------------------------------------------------
    # pgmpy 1.1.2 compatible approach
    #
    # Do NOT use:
    #
    # model.fit(
    #     df,
    #     estimator=BayesianEstimator,
    #     prior_type="BDeu",
    #     equivalent_sample_size=ESS
    # )
    #
    # In pgmpy 1.1.2 these arguments are handled through
    # BayesianEstimator.get_parameters().
    # --------------------------------------------------------

    estimator = BayesianEstimator(
        model,
        df,
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    # --------------------------------------------------------
    # Add learned CPDs to the model
    # --------------------------------------------------------

    model.add_cpds(
        *cpds
    )

    # --------------------------------------------------------
    # Verify model consistency
    # --------------------------------------------------------

    if not model.check_model():

        raise ValueError(
            "Learned Bayesian network "
            "failed consistency check."
        )

    print(
        "CPDs learned: "
        f"{len(cpds)}"
    )

    print(
        "Model consistency: PASSED"
    )

    return model


# ============================================================
# CPT SUMMARY
# ============================================================

def create_cpt_summary(
    model,
    df,
):

    records = []

    for node in VARIABLES:

        parents = list(
            model.predecessors(
                node
            )
        )

        states = int(
            df[node].nunique()
        )

        parent_configurations = 1

        for parent in parents:

            parent_configurations *= int(
                df[parent].nunique()
            )

        cpt_entries = (
            states
            * parent_configurations
        )

        cpd = model.get_cpds(
            node
        )

        records.append(
            {
                "variable": node,
                "parents": (
                    ", ".join(parents)
                    if parents
                    else "-"
                ),
                "number_of_parents":
                    len(parents),
                "states":
                    states,
                "parent_configurations":
                    parent_configurations,
                "cpt_entries":
                    cpt_entries,
                "cpd_available":
                    cpd is not None,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# SAVE CPT
# ============================================================

def save_cpt(
    model,
    node,
):

    cpd = model.get_cpds(
        node
    )

    if cpd is None:

        raise ValueError(
            f"No CPD found for node: "
            f"{node}"
        )

    values = cpd.get_values()

    parents = list(
        model.predecessors(
            node
        )
    )

    state_names = (
        cpd.state_names
    )

    rows = []

    # --------------------------------------------------------
    # ROOT NODE
    # --------------------------------------------------------

    if not parents:

        node_states = (
            state_names[node]
        )

        for row_index, state in enumerate(
            node_states
        ):

            rows.append(
                {
                    node: state,
                    "probability":
                        float(
                            values[
                                row_index,
                                0,
                            ]
                        ),
                }
            )

    # --------------------------------------------------------
    # NODE WITH PARENTS
    # --------------------------------------------------------

    else:

        parent_states = [
            state_names[parent]
            for parent in parents
        ]

        combinations = itertools.product(
            *parent_states
        )

        for column_index, combination in (
            enumerate(combinations)
        ):

            for row_index, node_state in (
                enumerate(
                    state_names[node]
                )
            ):

                record = {}

                for parent, parent_state in zip(
                    parents,
                    combination,
                ):

                    record[parent] = (
                        parent_state
                    )

                record[node] = (
                    node_state
                )

                record["probability"] = float(
                    values[
                        row_index,
                        column_index,
                    ]
                )

                rows.append(
                    record
                )

    cpt_df = pd.DataFrame(
        rows
    )

    output_path = (
        CPT_DIR
        / f"{node}_cpt.csv"
    )

    cpt_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# CPT VALIDATION
# ============================================================

def validate_cpts(
    model,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "CPT VALIDATION"
    )

    print(
        "=" * 70
    )

    records = []

    all_valid = True

    for node in VARIABLES:

        cpd = model.get_cpds(
            node
        )

        if cpd is None:

            all_valid = False

            print(
                f"{node}: FAILED "
                "(CPD missing)"
            )

            records.append(
                {
                    "variable": node,
                    "valid": False,
                    "minimum_probability":
                        None,
                    "maximum_probability":
                        None,
                    "minimum_column_sum":
                        None,
                    "maximum_column_sum":
                        None,
                }
            )

            continue

        values = cpd.get_values()

        minimum_probability = float(
            values.min()
        )

        maximum_probability = float(
            values.max()
        )

        column_sums = (
            values.sum(
                axis=0
            )
        )

        minimum_column_sum = float(
            column_sums.min()
        )

        maximum_column_sum = float(
            column_sums.max()
        )

        valid = (
            minimum_probability >= 0
            and maximum_probability <= 1
            and abs(
                minimum_column_sum - 1.0
            ) < 1e-8
            and abs(
                maximum_column_sum - 1.0
            ) < 1e-8
        )

        if not valid:
            all_valid = False

        status = (
            "PASSED"
            if valid
            else "FAILED"
        )

        print(
            f"{node}: {status} | "
            f"min={minimum_probability:.8f} | "
            f"max={maximum_probability:.8f} | "
            f"sum={minimum_column_sum:.8f}-"
            f"{maximum_column_sum:.8f}"
        )

        records.append(
            {
                "variable": node,
                "valid": valid,
                "minimum_probability":
                    minimum_probability,
                "maximum_probability":
                    maximum_probability,
                "minimum_column_sum":
                    minimum_column_sum,
                "maximum_column_sum":
                    maximum_column_sum,
            }
        )

    validation_df = pd.DataFrame(
        records
    )

    validation_path = (
        OUTPUT_DIR
        / "cpt_validation.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    return (
        all_valid,
        validation_df,
    )


# ============================================================
# CPT STATISTICS
# ============================================================

def calculate_cpt_statistics(
    model,
):

    total_entries = 0
    zero_entries = 0

    for node in VARIABLES:

        cpd = model.get_cpds(
            node
        )

        values = cpd.get_values()

        total_entries += values.size

        zero_entries += int(
            (values == 0).sum()
        )

    return {
        "total_cpt_entries":
            int(total_entries),
        "zero_probability_entries":
            int(zero_entries),
        "nonzero_probability_entries":
            int(
                total_entries
                - zero_entries
            ),
    }


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    df,
    edges,
    cpts_valid,
    cpt_statistics,
):

    metadata = {

        "project":
            "Probability-Reasoning",

        "dataset":
            "ACTG175",

        "phase":
            "Phase 9 - Parameter Learning",

        "representation":
            "sparse",

        "development_rows":
            len(df),

        "number_of_variables":
            len(VARIABLES),

        "number_of_edges":
            len(edges),

        "variables":
            VARIABLES,

        "target":
            TARGET,

        "estimator":
            "BayesianEstimator",

        "prior_type":
            "BDeu",

        "equivalent_sample_size":
            ESS,

        "dag_valid":
            True,

        "temporal_constraints_valid":
            True,

        "forbidden_label_to_trt":
            True,

        "cpts_valid":
            bool(cpts_valid),

        "cpt_statistics":
            cpt_statistics,

        "final_dag":
            [
                {
                    "source": source,
                    "target": target,
                }
                for source, target in sorted(
                    edges
                )
            ],
    }

    metadata_path = (
        OUTPUT_DIR
        / "parameter_metadata.json"
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

    return metadata_path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ACTG175 PHASE-9 PARAMETER LEARNING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    edges = load_final_dag()

    print(
        f"\nFinal DAG edges: "
        f"{len(edges)}"
    )

    print(
        "\nFinal DAG:"
    )

    for source, target in sorted(
        edges
    ):

        print(
            f"{source} -> {target}"
        )

    # --------------------------------------------------------
    # Validate DAG
    # --------------------------------------------------------

    model = validate_graph(
        edges,
        df,
    )

    print(
        "\nFinal DAG validation: PASSED"
    )

    # --------------------------------------------------------
    # Learn parameters
    # --------------------------------------------------------

    model = learn_parameters(
        model,
        df,
    )

    # --------------------------------------------------------
    # CPT summary
    # --------------------------------------------------------

    summary_df = (
        create_cpt_summary(
            model,
            df,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CPT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "parameter_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save individual CPTs
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING CPTs"
    )

    print(
        "=" * 70
    )

    for node in VARIABLES:

        output_path = save_cpt(
            model,
            node,
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Validate CPTs
    # --------------------------------------------------------

    (
        cpts_valid,
        _,
    ) = validate_cpts(
        model
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    cpt_statistics = (
        calculate_cpt_statistics(
            model
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CPT STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"Total CPT entries: "
        f"{cpt_statistics['total_cpt_entries']}"
    )

    print(
        f"Zero-probability entries: "
        f"{cpt_statistics['zero_probability_entries']}"
    )

    print(
        f"Non-zero probability entries: "
        f"{cpt_statistics['nonzero_probability_entries']}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = save_metadata(
        df=df,
        edges=edges,
        cpts_valid=cpts_valid,
        cpt_statistics=cpt_statistics,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE-9 PARAMETER LEARNING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCPT validation: "
        f"{'PASSED' if cpts_valid else 'FAILED'}"
    )

    print(
        f"\nCPT directory:"
        f"\n{CPT_DIR}"
    )

    print(
        f"\nParameter summary:"
        f"\n{summary_path}"
    )

    print(
        f"\nCPT validation:"
        f"\n{OUTPUT_DIR / 'cpt_validation.csv'}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_path}"
    )

    if not cpts_valid:

        raise RuntimeError(
            "One or more CPTs failed validation."
        )

    print(
        "\nAll parameter-learning "
        "checks PASSED."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()