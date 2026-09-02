import itertools
import json
import shutil
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

OLD_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

ALTERNATIVE_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag"
    / "final_dag_edges.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "final_model"
)

FINAL_DAG_DIR = (
    OUTPUT_ROOT
    / "dag"
)

FINAL_CPT_DIR = (
    OUTPUT_ROOT
    / "cpts"
)

FINAL_METADATA_DIR = (
    OUTPUT_ROOT
    / "metadata"
)

BASELINE_BACKUP_DIR = (
    OUTPUT_ROOT
    / "baseline_22_edge"
)

for directory in [
    OUTPUT_ROOT,
    FINAL_DAG_DIR,
    FINAL_CPT_DIR,
    FINAL_METADATA_DIR,
    BASELINE_BACKUP_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# VARIABLES
# ============================================================

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

TREATMENT = "trt"
TARGET = "label"

ALL_VARIABLES = (
    BASELINE_VARIABLES
    + [
        TREATMENT,
        TARGET,
    ]
)

ESS = 10


# ============================================================
# PRINTING
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATA
# ============================================================

def load_development_data():

    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DEVELOPMENT_PATH}"
        )

    data = pd.read_csv(
        DEVELOPMENT_PATH
    )

    missing_columns = (
        set(ALL_VARIABLES)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Development dataset is missing variables:\n"
            f"{sorted(missing_columns)}"
        )

    data = data[
        ALL_VARIABLES
    ].copy()

    # All variables in the selected representation
    # are treated as discrete states.
    for column in ALL_VARIABLES:
        data[column] = (
            data[column]
            .astype(str)
        )

    return data


# ============================================================
# DAG LOCATION
# ============================================================

def locate_old_dag():

    candidates = [
        OLD_DAG_PATH,
        ALTERNATIVE_DAG_PATH,
    ]

    discovered = list(
        (
            PROJECT_ROOT
            / "results"
            / "structure_learning"
            / "final"
        ).glob(
            "**/*dag*edges.csv"
        )
    )

    candidates.extend(
        discovered
    )

    # Remove duplicates while preserving order.
    unique_candidates = []

    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)

    for path in unique_candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate the previous final DAG.\n\n"
        f"Expected:\n{OLD_DAG_PATH}\n\n"
        f"Alternative:\n{ALTERNATIVE_DAG_PATH}"
    )


# ============================================================
# EDGE LOADING
# ============================================================

def load_edges(path):

    edges_df = pd.read_csv(
        path
    )

    required_columns = {
        "source",
        "target",
    }

    if not required_columns.issubset(
        edges_df.columns
    ):
        raise ValueError(
            "DAG edge file must contain "
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
            (
                source,
                target,
            )
        )

    return edges


# ============================================================
# GRAPH
# ============================================================

def build_graph(edges):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        ALL_VARIABLES
    )

    graph.add_edges_from(
        edges
    )

    return graph


def validate_graph(edges):

    graph = build_graph(
        edges
    )

    # --------------------------------------------------------
    # DAG check
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(
        graph
    ):
        raise ValueError(
            "Final graph is not a DAG."
        )

    print(
        "DAG structure: PASSED"
    )

    # --------------------------------------------------------
    # Forbidden edge
    # --------------------------------------------------------

    forbidden_edge = (
        TARGET,
        TREATMENT,
    )

    if forbidden_edge in edges:
        raise ValueError(
            "Forbidden edge detected: "
            "label -> trt"
        )

    print(
        "Forbidden label -> trt: PASSED"
    )

    # --------------------------------------------------------
    # Outcome must not have outgoing edges
    # --------------------------------------------------------

    outgoing_from_label = [
        edge
        for edge in edges
        if edge[0] == TARGET
    ]

    if outgoing_from_label:
        raise ValueError(
            "Outcome has outgoing edges:\n"
            f"{outgoing_from_label}"
        )

    print(
        "Outcome direction: PASSED"
    )

    # --------------------------------------------------------
    # Treatment -> outcome must exist
    # --------------------------------------------------------

    treatment_edge = (
        TREATMENT,
        TARGET,
    )

    if treatment_edge not in edges:
        raise ValueError(
            "Required edge missing: "
            "trt -> label"
        )

    print(
        "Treatment -> outcome edge: PASSED"
    )

    return graph


# ============================================================
# FINAL EDGE CREATION
# ============================================================

def add_treatment_edge(edges):

    final_edges = list(
        edges
    )

    treatment_edge = (
        TREATMENT,
        TARGET,
    )

    if treatment_edge not in final_edges:
        final_edges.append(
            treatment_edge
        )

    return final_edges


# ============================================================
# PARAMETER LEARNING
# ============================================================

def learn_bn(
    data,
    edges,
):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        ALL_VARIABLES
    )

    model.add_edges_from(
        edges
    )

    estimator = BayesianEstimator(
        model,
        data,
    )

    # IMPORTANT:
    #
    # pgmpy 1.1.2 does not support passing
    # prior_type/equivalent_sample_size
    # directly to model.fit().
    #
    # get_parameters() is used instead.

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    model.add_cpds(
        *cpds
    )

    if not model.check_model():
        raise ValueError(
            "Bayesian network consistency "
            "check failed."
        )

    return model


# ============================================================
# CPT SUMMARY
# ============================================================

def get_cpt_summary(model):

    rows = []

    for variable in ALL_VARIABLES:

        cpd = model.get_cpds(
            variable
        )

        if cpd is None:

            rows.append(
                {
                    "variable": variable,
                    "parents": "",
                    "number_of_parents": 0,
                    "states": 0,
                    "parent_configurations": 0,
                    "cpt_entries": 0,
                    "cpd_available": False,
                }
            )

            continue

        # Use the graph to identify parents.
        parents = list(
            model.predecessors(
                variable
            )
        )

        state_count = len(
            cpd.state_names[
                variable
            ]
        )

        parent_configurations = 1

        for parent in parents:

            parent_states = len(
                cpd.state_names.get(
                    parent,
                    [],
                )
            )

            parent_configurations *= (
                parent_states
            )

        entries = (
            state_count
            * parent_configurations
        )

        rows.append(
            {
                "variable": variable,
                "parents": (
                    ", ".join(parents)
                    if parents
                    else "-"
                ),
                "number_of_parents": len(
                    parents
                ),
                "states": state_count,
                "parent_configurations": (
                    parent_configurations
                ),
                "cpt_entries": entries,
                "cpd_available": True,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CPT VALIDATION
# ============================================================

def validate_cpds(model):

    validation_rows = []

    total_entries = 0
    zero_probability_entries = 0

    for variable in ALL_VARIABLES:

        cpd = model.get_cpds(
            variable
        )

        if cpd is None:

            validation_rows.append(
                {
                    "variable": variable,
                    "status": "FAILED",
                    "minimum_probability": np.nan,
                    "maximum_probability": np.nan,
                    "minimum_column_sum": np.nan,
                    "maximum_column_sum": np.nan,
                    "zero_probability_entries": np.nan,
                }
            )

            continue

        values = np.asarray(
            cpd.values,
            dtype=float,
        )

        total_entries += (
            values.size
        )

        zero_count = int(
            np.sum(
                values <= 0
            )
        )

        zero_probability_entries += (
            zero_count
        )

        minimum_probability = float(
            values.min()
        )

        maximum_probability = float(
            values.max()
        )

        # Every CPD is represented as:
        #
        # child_state x parent_configuration
        #
        # after flattening the parent dimensions.
        #
        if values.ndim == 1:

            column_sums = np.array(
                [
                    values.sum()
                ]
            )

        else:

            column_sums = (
                values.reshape(
                    values.shape[0],
                    -1,
                )
                .sum(axis=0)
            )

        minimum_column_sum = float(
            column_sums.min()
        )

        maximum_column_sum = float(
            column_sums.max()
        )

        passed = (
            np.allclose(
                column_sums,
                1.0,
                atol=1e-8,
            )
            and np.all(
                np.isfinite(values)
            )
            and np.all(
                values >= 0
            )
            and np.all(
                values <= 1
            )
        )

        validation_rows.append(
            {
                "variable": variable,
                "status": (
                    "PASSED"
                    if passed
                    else "FAILED"
                ),
                "minimum_probability": (
                    minimum_probability
                ),
                "maximum_probability": (
                    maximum_probability
                ),
                "minimum_column_sum": (
                    minimum_column_sum
                ),
                "maximum_column_sum": (
                    maximum_column_sum
                ),
                "zero_probability_entries": (
                    zero_count
                ),
            }
        )

        if not passed:
            raise ValueError(
                f"CPT validation failed "
                f"for variable: {variable}"
            )

    validation_df = pd.DataFrame(
        validation_rows
    )

    return (
        validation_df,
        total_entries,
        zero_probability_entries,
    )


# ============================================================
# CPT EXPORT
# ============================================================

def save_cpts(model):
    """
    Save all CPDs as readable long-format CSV files.

    IMPORTANT:
    pgmpy can store multidimensional CPDs in an axis order
    different from the order returned by model.predecessors().

    Therefore this function uses:

        cpd.variables

    as the authoritative dimension ordering.

    Example:

        P(label | cd40, z30, trt)

    may internally be stored as:

        ['label', 'cd40', 'trt', 'z30']

    This function handles that automatically.
    """

    saved = []

    for variable in ALL_VARIABLES:

        cpd = model.get_cpds(
            variable
        )

        if cpd is None:
            raise ValueError(
                f"No CPD found for variable: "
                f"{variable}"
            )

        # ----------------------------------------------------
        # ACTUAL pgmpy CPD ORDER
        # ----------------------------------------------------

        cpd_variables = list(
            cpd.variables
        )

        if not cpd_variables:
            raise ValueError(
                f"CPD has no variables: "
                f"{variable}"
            )

        if cpd_variables[0] != variable:
            raise ValueError(
                f"Unexpected CPD variable ordering "
                f"for {variable}:\n"
                f"{cpd_variables}"
            )

        parent_variables = (
            cpd_variables[1:]
        )

        # ----------------------------------------------------
        # CPD state names
        # ----------------------------------------------------

        state_names = (
            cpd.state_names
        )

        variable_states = list(
            state_names[
                variable
            ]
        )

        # ----------------------------------------------------
        # CPD numerical array
        # ----------------------------------------------------

        values = np.asarray(
            cpd.values,
            dtype=float,
        )

        # ----------------------------------------------------
        # Expected dimensions based on pgmpy's own CPD
        # ----------------------------------------------------

        expected_shape = tuple(
            int(cardinality)
            for cardinality in (
                cpd.cardinality
            )
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected CPD shape for "
                f"{variable}.\n"
                f"CPD variables: "
                f"{cpd_variables}\n"
                f"Expected shape: "
                f"{expected_shape}\n"
                f"Actual shape: "
                f"{values.shape}"
            )

        rows = []

        # ====================================================
        # ROOT VARIABLE
        # ====================================================

        if len(parent_variables) == 0:

            values_flat = (
                values.reshape(-1)
            )

            if len(values_flat) != len(
                variable_states
            ):
                raise ValueError(
                    f"Root CPD size mismatch "
                    f"for {variable}."
                )

            for state_index, state in enumerate(
                variable_states
            ):

                rows.append(
                    {
                        variable: state,
                        "probability": float(
                            values_flat[
                                state_index
                            ]
                        ),
                    }
                )

        # ====================================================
        # VARIABLE WITH PARENTS
        # ====================================================

        else:

            # ------------------------------------------------
            # Parent state lists in ACTUAL CPD ORDER.
            # ------------------------------------------------

            parent_state_lists = [
                list(
                    state_names[
                        parent
                    ]
                )
                for parent in parent_variables
            ]

            parent_combinations = itertools.product(
                *parent_state_lists
            )

            for parent_combination in (
                parent_combinations
            ):

                # --------------------------------------------
                # Convert parent state names into axis indexes.
                # --------------------------------------------

                parent_indices = []

                for parent_index, parent in enumerate(
                    parent_variables
                ):

                    parent_states = list(
                        state_names[
                            parent
                        ]
                    )

                    selected_state = (
                        parent_combination[
                            parent_index
                        ]
                    )

                    state_index = (
                        parent_states.index(
                            selected_state
                        )
                    )

                    parent_indices.append(
                        state_index
                    )

                # --------------------------------------------
                # Build multidimensional index.
                #
                # Axis 0 = child variable.
                # Axis 1+ = parents in cpd.variables order.
                # --------------------------------------------

                multidimensional_index = (
                    (
                        slice(None),
                        *parent_indices,
                    )
                )

                probability_vector = (
                    values[
                        multidimensional_index
                    ]
                )

                probability_vector = (
                    np.asarray(
                        probability_vector,
                        dtype=float,
                    )
                    .reshape(-1)
                )

                if len(
                    probability_vector
                ) != len(
                    variable_states
                ):
                    raise ValueError(
                        f"Probability vector size "
                        f"mismatch for {variable}.\n"
                        f"CPD variables: "
                        f"{cpd_variables}\n"
                        f"Parent combination: "
                        f"{parent_combination}\n"
                        f"Probability vector size: "
                        f"{len(probability_vector)}\n"
                        f"Expected: "
                        f"{len(variable_states)}"
                    )

                # --------------------------------------------
                # Create one CSV row per child state.
                # --------------------------------------------

                for state_index, state in enumerate(
                    variable_states
                ):

                    row = {}

                    for parent_index, parent in enumerate(
                        parent_variables
                    ):

                        row[parent] = (
                            parent_combination[
                                parent_index
                            ]
                        )

                    row[variable] = state

                    row["probability"] = float(
                        probability_vector[
                            state_index
                        ]
                    )

                    rows.append(
                        row
                    )

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        cpt_df = pd.DataFrame(
            rows
        )

        if cpt_df.empty:
            raise ValueError(
                f"Empty CPT generated "
                f"for {variable}"
            )

        # ----------------------------------------------------
        # Clean column order.
        # ----------------------------------------------------

        ordered_columns = (
            parent_variables
            + [
                variable,
                "probability",
            ]
        )

        cpt_df = cpt_df[
            ordered_columns
        ]

        # ----------------------------------------------------
        # Export validation.
        # ----------------------------------------------------

        probabilities = (
            cpt_df[
                "probability"
            ].to_numpy(
                dtype=float
            )
        )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise ValueError(
                f"Non-finite probability "
                f"found for {variable}"
            )

        if np.any(
            probabilities < 0
        ):
            raise ValueError(
                f"Negative probability "
                f"found for {variable}"
            )

        if np.any(
            probabilities > 1
        ):
            raise ValueError(
                f"Probability greater than 1 "
                f"found for {variable}"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            FINAL_CPT_DIR
            / f"{variable}_cpt.csv"
        )

        cpt_df.to_csv(
            output_path,
            index=False,
        )

        saved.append(
            str(output_path)
        )

    return saved


# ============================================================
# SAVE DAG
# ============================================================

def save_dag(edges):

    dag_df = pd.DataFrame(
        edges,
        columns=[
            "source",
            "target",
        ],
    )

    output_path = (
        FINAL_DAG_DIR
        / "final_dag_edges.csv"
    )

    dag_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# SAVE DOT GRAPH
# ============================================================

def save_dag_dot(edges):

    output_path = (
        FINAL_DAG_DIR
        / "final_dag.dot"
    )

    lines = ['digraph "ACTG175 Final Bayesian Network" {']
    for source, target in sorted(edges):
        lines.append(f"    {source} -> {target};")
    lines.append("}\n")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return output_path


# ============================================================
# PRESERVE BASELINE
# ============================================================

def copy_baseline_dag(
    old_dag_path
):

    destination = (
        BASELINE_BACKUP_DIR
        / "baseline_final_dag_edges.csv"
    )

    # Never overwrite the original baseline.
    if not destination.exists():

        shutil.copy2(
            old_dag_path,
            destination,
        )

    return destination


# ============================================================
# MODEL STATISTICS
# ============================================================

def calculate_statistics(
    edges,
    model,
    cpt_summary,
):

    total_cpt_entries = int(
        cpt_summary[
            "cpt_entries"
        ].sum()
    )

    maximum_indegree = 0

    for node in model.nodes():

        indegree = len(
            list(
                model.predecessors(
                    node
                )
            )
        )

        maximum_indegree = max(
            maximum_indegree,
            indegree,
        )

    return {
        "number_of_nodes": len(
                model.nodes()
            ),
        "number_of_edges": len(edges),
        "maximum_indegree": int(
            maximum_indegree
        ),
        "total_cpt_entries": int(
            total_cpt_entries
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ACTG175 PHASE-15"
    )

    print(
        "FINAL BAYESIAN NETWORK REBUILD"
    )

    print(
        "\nDecision from Phase 14:"
    )

    print(
        "ACCEPT_TREATMENT_EDGE"
    )

    print(
        "\nPromoting:"
    )

    print(
        "trt -> label"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The original 22-edge model is preserved "
        "as the baseline."
    )

    # ========================================================
    # LOAD DEVELOPMENT DATA
    # ========================================================

    data = load_development_data()

    print(
        f"\nDevelopment dataset:"
        f"\n{DEVELOPMENT_PATH}"
    )

    print(
        f"Shape: {data.shape}"
    )

    missing_values = int(
        data.isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values: "
        f"{missing_values}"
    )

    duplicate_rows = int(
        data.duplicated()
        .sum()
    )

    if duplicate_rows > 0:

        print(
            f"Duplicate rows: "
            f"{duplicate_rows}"
        )

    # ========================================================
    # LOAD PREVIOUS DAG
    # ========================================================

    old_dag_path = (
        locate_old_dag()
    )

    print(
        f"\nPrevious final DAG:"
        f"\n{old_dag_path}"
    )

    old_edges = load_edges(
        old_dag_path
    )

    print(
        f"Previous edges: "
        f"{len(old_edges)}"
    )

    # ========================================================
    # PRESERVE BASELINE
    # ========================================================

    baseline_backup = (
        copy_baseline_dag(
            old_dag_path
        )
    )

    print(
        f"Baseline DAG preserved:"
        f"\n{baseline_backup}"
    )

    # ========================================================
    # CREATE FINAL DAG
    # ========================================================

    final_edges = (
        add_treatment_edge(
            old_edges
        )
    )

    print_header(
        "FINAL DAG"
    )

    print(
        f"Previous edges: "
        f"{len(old_edges)}"
    )

    print(
        f"Final edges:    "
        f"{len(final_edges)}"
    )

    print(
        "\nNew edge:"
    )

    print(
        "trt -> label"
    )

    print(
        "\nFinal DAG edges:"
    )

    for source, target in sorted(
        final_edges
    ):

        print(
            f"{source} -> {target}"
        )

    # ========================================================
    # VALIDATE DAG
    # ========================================================

    print_header(
        "FINAL DAG VALIDATION"
    )

    validate_graph(
        final_edges
    )

    print(
        "\nFinal DAG validation: PASSED"
    )

    # ========================================================
    # PARAMETER LEARNING
    # ========================================================

    print_header(
        "FINAL PARAMETER LEARNING"
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
        "\nLearning final CPTs..."
    )

    model = learn_bn(
        data,
        final_edges,
    )

    cpds = model.get_cpds()

    print(
        f"CPDs learned: "
        f"{len(cpds)}"
    )

    print(
        "Model consistency: PASSED"
    )

    # ========================================================
    # CPT SUMMARY
    # ========================================================

    cpt_summary = (
        get_cpt_summary(
            model
        )
    )

    print(
        "\nCPT summary:"
    )

    print(
        cpt_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # CPT VALIDATION
    # ========================================================

    print_header(
        "CPT VALIDATION"
    )

    (
        validation_df,
        total_entries,
        zero_probability_entries,
    ) = validate_cpds(
        model
    )

    print(
        validation_df.to_string(
            index=False
        )
    )

    print(
        f"\nTotal CPT entries: "
        f"{total_entries}"
    )

    print(
        f"Zero-probability entries: "
        f"{zero_probability_entries}"
    )

    if (
        validation_df[
            "status"
        ]
        != "PASSED"
    ).any():

        raise ValueError(
            "At least one CPT failed validation."
        )

    print(
        "\nCPT validation: PASSED"
    )

    # ========================================================
    # SAVE FINAL CPTS
    # ========================================================

    print_header(
        "SAVING FINAL MODEL"
    )

    saved_cpts = save_cpts(
        model
    )

    for path in saved_cpts:

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # SAVE FINAL DAG
    # ========================================================

    final_dag_path = (
        save_dag(
            final_edges
        )
    )

    print(
        f"\nFinal DAG saved:"
        f"\n{final_dag_path}"
    )

    # ========================================================
    # SAVE DOT
    # ========================================================

    dot_path = save_dag_dot(
        final_edges
    )

    if dot_path is not None:

        print(
            f"Graph DOT file saved:"
            f"\n{dot_path}"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    statistics = (
        calculate_statistics(
            final_edges,
            model,
            cpt_summary,
        )
    )

    print_header(
        "FINAL MODEL STATISTICS"
    )

    print(
        f"Nodes:            "
        f"{statistics['number_of_nodes']}"
    )

    print(
        f"Edges:            "
        f"{statistics['number_of_edges']}"
    )

    print(
        f"Maximum indegree: "
        f"{statistics['maximum_indegree']}"
    )

    print(
        f"Total CPT entries: "
        f"{statistics['total_cpt_entries']}"
    )

    # ========================================================
    # SAVE SUMMARIES
    # ========================================================

    cpt_summary_path = (
        FINAL_METADATA_DIR
        / "final_cpt_summary.csv"
    )

    cpt_summary.to_csv(
        cpt_summary_path,
        index=False,
    )

    validation_path = (
        FINAL_METADATA_DIR
        / "final_cpt_validation.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    statistics_path = (
        FINAL_METADATA_DIR
        / "final_model_statistics.json"
    )

    with open(
        statistics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4,
        )

    metadata = {
        "phase": 15,
        "model": (
            "ACTG175 Final Bayesian Network"
        ),
        "representation": "sparse",
        "training_dataset": str(
            DEVELOPMENT_PATH
        ),
        "training_rows": len(data),
        "test_data_used": False,
        "estimator": (
            "BayesianEstimator"
        ),
        "prior": "BDeu",
        "equivalent_sample_size": ESS,
        "previous_edge_count": len(old_edges),
        "final_edge_count": len(final_edges),
        "new_edge": "trt -> label",
        "dag_validation": "PASSED",
        "model_consistency": "PASSED",
        "cpt_validation": "PASSED",
        "total_cpt_entries": int(
            total_entries
        ),
        "zero_probability_entries": int(
            zero_probability_entries
        ),
        "baseline_dag_preserved": True,
        "final_dag_path": str(
            final_dag_path
        ),
        "cpt_directory": str(
            FINAL_CPT_DIR
        ),
        "dot_graph": (
            str(dot_path)
            if dot_path is not None
            else None
        ),
    }

    metadata_path = (
        FINAL_METADATA_DIR
        / "final_model_metadata.json"
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

    # ========================================================
    # OUTPUT LOCATIONS
    # ========================================================

    print(
        f"\nCPT summary:"
        f"\n{cpt_summary_path}"
    )

    print(
        f"\nCPT validation:"
        f"\n{validation_path}"
    )

    print(
        f"\nStatistics:"
        f"\n{statistics_path}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_path}"
    )

    # ========================================================
    # FINAL CONFIRMATION
    # ========================================================

    print_header(
        "PHASE-15 RESULT"
    )

    print(
        "Final DAG: PASSED"
    )

    print(
        "Final parameter learning: PASSED"
    )

    print(
        "Model consistency: PASSED"
    )

    print(
        "CPT validation: PASSED"
    )

    print(
        "\nFinal model contains:"
    )

    print(
        "trt -> label"
    )

    print(
        "\nFinal edge count: 23"
    )

    print(
        "\nTotal CPT entries: "
        f"{total_entries}"
    )

    print(
        "\nZero-probability entries: "
        f"{zero_probability_entries}"
    )

    print(
        "\nThe original 22-edge DAG was preserved "
        "as the baseline."
    )

    print(
        "\nPHASE-15 COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()