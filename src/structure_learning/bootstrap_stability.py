from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from pgmpy.causal_discovery import (
    HillClimbSearch,
    ExpertKnowledge,
)


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "sparse"
    / "bootstrap"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

N_BOOTSTRAPS = 100

RANDOM_SEED = 42

MAX_INDEGREE = 3

SCORING_METHOD = "bic-d"

RETURN_TYPE = "dag"

SHOW_PROGRESS = False


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


# ============================================================
# TEMPORAL GROUPS
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

TREATMENT_VARIABLES = [
    "trt"
]

OUTCOME_VARIABLES = [
    "label"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    if df.columns.tolist() != VARIABLES:

        raise ValueError(
            "Unexpected dataset schema.\n\n"
            f"Expected:\n{VARIABLES}\n\n"
            f"Found:\n{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = int(
        df.isnull().sum().sum()
    )

    if missing != 0:

        raise ValueError(
            f"Dataset contains "
            f"{missing} missing values."
        )

    return df


# ============================================================
# CONVERT ALL VARIABLES TO DISCRETE STATES
# ============================================================

def convert_to_discrete(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    for column in VARIABLES:

        result[column] = (
            result[column]
            .astype(str)
        )

    return result


# ============================================================
# BUILD EXPERT KNOWLEDGE
# ============================================================

def build_expert_knowledge():

    temporal_order = [
        BASELINE_VARIABLES,
        TREATMENT_VARIABLES,
        OUTCOME_VARIABLES,
    ]

    forbidden_edges = {
        (
            "label",
            "trt",
        )
    }

    knowledge = ExpertKnowledge(
        forbidden_edges=forbidden_edges,
        temporal_order=temporal_order,
    )

    return knowledge


# ============================================================
# GRAPH VALIDATION
# ============================================================

def validate_graph(
    model,
):

    tier = {}

    # Baseline = tier 0
    for variable in BASELINE_VARIABLES:
        tier[variable] = 0

    # Treatment = tier 1
    for variable in TREATMENT_VARIABLES:
        tier[variable] = 1

    # Outcome = tier 2
    for variable in OUTCOME_VARIABLES:
        tier[variable] = 2

    violations = []

    for source, target in model.edges():

        if tier[source] > tier[target]:

            violations.append(
                (
                    source,
                    target,
                )
            )

    return violations


# ============================================================
# LEARN ONE BOOTSTRAP GRAPH
# ============================================================

def learn_bootstrap_graph(
    sample: pd.DataFrame,
    expert_knowledge: ExpertKnowledge,
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # In pgmpy causal_discovery.HillClimbSearch,
    # configuration belongs in the constructor.
    # .fit() receives the data.
    # --------------------------------------------------------

    search = HillClimbSearch(
        scoring_method=SCORING_METHOD,
        max_indegree=MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        return_type=RETURN_TYPE,
        show_progress=SHOW_PROGRESS,
    )

    fitted = search.fit(
        sample
    )

    # --------------------------------------------------------
    # Get learned DAG
    # --------------------------------------------------------

    model = fitted.causal_graph_

    # --------------------------------------------------------
    # Validate temporal constraints
    # --------------------------------------------------------

    violations = validate_graph(
        model
    )

    if violations:

        raise RuntimeError(
            "Constraint violation detected:\n"
            + "\n".join(
                f"{source} -> {target}"
                for source, target
                in violations
            )
        )

    return model


# ============================================================
# GRAPH STATISTICS
# ============================================================

def calculate_graph_statistics(
    model,
    data,
):

    edges = list(
        model.edges()
    )

    indegrees = [
        model.in_degree(node)
        for node in model.nodes()
    ]

    maximum_indegree = max(
        indegrees
    )

    total_cpt_entries = 0

    for node in model.nodes():

        node_states = (
            data[node].nunique()
        )

        parents = list(
            model.get_parents(node)
        )

        parent_configurations = 1

        for parent in parents:

            parent_configurations *= (
                data[parent].nunique()
            )

        total_cpt_entries += (
            node_states
            * parent_configurations
        )

    return {
        "edges": len(edges),
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

    print(
        "=" * 70
    )

    print(
        "ACTG175 BOOTSTRAP STRUCTURE STABILITY"
    )

    print(
        "=" * 70
    )

    print(
        "\nConfiguration:"
    )

    print(
        f"Development dataset: "
        f"{DATA_PATH}"
    )

    print(
        f"Bootstrap replicates: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print(
        f"Maximum indegree: "
        f"{MAX_INDEGREE}"
    )

    print(
        f"Scoring method: "
        f"{SCORING_METHOD}"
    )

    print(
        "\nAlgorithm:"
    )

    print(
        "Constrained Hill Climbing"
    )

    print(
        "\nTest set:"
    )

    print(
        "NOT USED"
    )

    # ========================================================
    # LOAD ORIGINAL DATA
    # ========================================================

    data = load_data()

    print(
        "\nOriginal development shape:"
    )

    print(
        data.shape
    )

    # ========================================================
    # CONVERT TO DISCRETE
    # ========================================================

    data = convert_to_discrete(
        data
    )

    print(
        "\nData representation:"
    )

    print(
        "All 17 variables converted "
        "to discrete string states."
    )

    # ========================================================
    # EXPERT KNOWLEDGE
    # ========================================================

    expert_knowledge = (
        build_expert_knowledge()
    )

    print(
        "\nExpert knowledge:"
    )

    print(
        expert_knowledge
    )

    # ========================================================
    # RANDOM GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # EDGE COUNTS
    # ========================================================

    edge_counts = {}

    graph_records = []

    # ========================================================
    # BOOTSTRAP LOOP
    # ========================================================

    total_start = time.time()

    for bootstrap_id in range(
        1,
        N_BOOTSTRAPS + 1,
    ):

        print(
            f"\nBootstrap "
            f"{bootstrap_id}/{N_BOOTSTRAPS}"
        )

        iteration_start = (
            time.time()
        )

        # ----------------------------------------------------
        # Bootstrap sample
        # ----------------------------------------------------

        indices = rng.choice(
            len(data),
            size=len(data),
            replace=True,
        )

        sample = (
            data.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Learn graph
        # ----------------------------------------------------

        model = learn_bootstrap_graph(
            sample,
            expert_knowledge,
        )

        edges = list(
            model.edges()
        )

        print(
            f"Edges learned: "
            f"{len(edges)}"
        )

        # ----------------------------------------------------
        # Count directed edges
        # ----------------------------------------------------

        for source, target in edges:

            key = (
                source,
                target,
            )

            edge_counts[key] = (
                edge_counts.get(
                    key,
                    0,
                )
                + 1
            )

        # ----------------------------------------------------
        # Graph statistics
        # ----------------------------------------------------

        stats = calculate_graph_statistics(
            model,
            sample,
        )

        elapsed = (
            time.time()
            - iteration_start
        )

        graph_records.append(
            {
                "bootstrap": (
                    bootstrap_id
                ),
                "edges": (
                    stats["edges"]
                ),
                "maximum_indegree": (
                    stats[
                        "maximum_indegree"
                    ]
                ),
                "total_cpt_entries": (
                    stats[
                        "total_cpt_entries"
                    ]
                ),
                "elapsed_seconds": (
                    elapsed
                ),
            }
        )

        print(
            f"Time: "
            f"{elapsed:.2f}s"
        )

    # ========================================================
    # EDGE STABILITY
    # ========================================================

    stability_records = []

    for (
        source,
        target,
    ), count in edge_counts.items():

        stability = (
            count
            / N_BOOTSTRAPS
        )

        stability_records.append(
            {
                "source": source,
                "target": target,
                "count": int(
                    count
                ),
                "stability": float(
                    stability
                ),
                "stability_percentage": float(
                    stability * 100
                ),
            }
        )

    stability_df = pd.DataFrame(
        stability_records
    )

    if not stability_df.empty:

        stability_df = (
            stability_df
            .sort_values(
                [
                    "stability",
                    "source",
                    "target",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # SAVE ALL EDGE STABILITY
    # ========================================================

    all_edges_path = (
        RESULTS_DIR
        / "edge_stability.csv"
    )

    stability_df.to_csv(
        all_edges_path,
        index=False,
    )

    # ========================================================
    # >= 50% STABLE EDGES
    # ========================================================

    stable_df = (
        stability_df[
            stability_df[
                "stability"
            ]
            >= 0.50
        ]
        .copy()
    )

    stable_path = (
        RESULTS_DIR
        / "stable_edges.csv"
    )

    stable_df.to_csv(
        stable_path,
        index=False,
    )

    # ========================================================
    # >= 75% VERY STABLE EDGES
    # ========================================================

    very_stable_df = (
        stability_df[
            stability_df[
                "stability"
            ]
            >= 0.75
        ]
        .copy()
    )

    very_stable_path = (
        RESULTS_DIR
        / "very_stable_edges.csv"
    )

    very_stable_df.to_csv(
        very_stable_path,
        index=False,
    )

    # ========================================================
    # GRAPH STATISTICS
    # ========================================================

    graph_df = pd.DataFrame(
        graph_records
    )

    graph_stats_path = (
        RESULTS_DIR
        / "bootstrap_graph_statistics.csv"
    )

    graph_df.to_csv(
        graph_stats_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_elapsed = (
        time.time()
        - total_start
    )

    summary = {
        "dataset": "ACTG175",

        "representation": "sparse",

        "algorithm": (
            "Constrained Hill Climbing"
        ),

        "bootstrap_replicates": (
            N_BOOTSTRAPS
        ),

        "random_seed": (
            RANDOM_SEED
        ),

        "development_rows": (
            len(data)
        ),

        "variables": (
            len(VARIABLES)
        ),

        "maximum_indegree": (
            MAX_INDEGREE
        ),

        "scoring_method": (
            SCORING_METHOD
        ),

        "test_set_used": False,

        "unique_edges_observed": (
            len(stability_df)
        ),

        "stable_edges_50_percent": (
            len(stable_df)
        ),

        "stable_edges_75_percent": (
            len(very_stable_df)
        ),

        "mean_edges_per_bootstrap": (
            float(
                graph_df[
                    "edges"
                ].mean()
            )
        ),

        "std_edges_per_bootstrap": (
            float(
                graph_df[
                    "edges"
                ].std()
            )
        ),

        "mean_cpt_entries": (
            float(
                graph_df[
                    "total_cpt_entries"
                ].mean()
            )
        ),

        "total_elapsed_seconds": (
            float(
                total_elapsed
            )
        ),
    }

    summary_path = (
        RESULTS_DIR
        / "bootstrap_summary.json"
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

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BOOTSTRAP STABILITY RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBootstrap replicates: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Unique directed edges observed: "
        f"{len(stability_df)}"
    )

    print(
        f"Edges with stability >= 50%: "
        f"{len(stable_df)}"
    )

    print(
        f"Edges with stability >= 75%: "
        f"{len(very_stable_df)}"
    )

    print(
        f"Mean edges per bootstrap: "
        f"{graph_df['edges'].mean():.2f}"
    )

    print(
        f"Std edges per bootstrap: "
        f"{graph_df['edges'].std():.2f}"
    )

    # ========================================================
    # TOP STABLE EDGES
    # ========================================================

    print(
        "\nTop 25 most stable edges:"
    )

    if stability_df.empty:

        print(
            "No edges recorded."
        )

    else:

        print(
            stability_df[
                [
                    "source",
                    "target",
                    "count",
                    "stability_percentage",
                ]
            ]
            .head(25)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE PATHS
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        all_edges_path
    )

    print(
        stable_path
    )

    print(
        very_stable_path
    )

    print(
        graph_stats_path
    )

    print(
        summary_path
    )

    print(
        f"\nTotal runtime: "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BOOTSTRAP STABILITY COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()