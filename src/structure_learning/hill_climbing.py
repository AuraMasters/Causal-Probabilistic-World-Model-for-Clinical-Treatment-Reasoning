from pathlib import Path
import json
import sys

import pandas as pd

from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.structure_score import BIC, BDeu


# ============================================================
# Make local structure_learning modules importable
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from constraints import build_expert_knowledge


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

CANDIDATES = {
    "sparse": PROCESSED_DIR / "sparse",
    "rich": PROCESSED_DIR / "rich",
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Variables
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
# Configuration
# ============================================================

MAX_INDEGREE = 3

BDEU_EQUIVALENT_SAMPLE_SIZE = 10

MAX_ITER = 1_000_000

EPSILON = 1e-4

SHOW_PROGRESS = True


# ============================================================
# Load candidate data
# ============================================================

def load_candidate(
    name: str,
) -> pd.DataFrame:
    """
    Load one discretized development dataset.

    Only the development set is used for structure learning.
    The test set remains untouched.
    """

    path = (
        CANDIDATES[name]
        / "development.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    if df.columns.tolist() != VARIABLES:
        raise ValueError(
            f"Unexpected schema for {name}.\n\n"
            f"Expected:\n{VARIABLES}\n\n"
            f"Found:\n{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    missing = int(
        df.isnull().sum().sum()
    )

    if missing != 0:
        raise ValueError(
            f"{name} contains "
            f"{missing} missing values."
        )

    # --------------------------------------------------------
    # Convert states to categorical strings
    # --------------------------------------------------------

    for column in VARIABLES:
        df[column] = (
            df[column]
            .astype(str)
        )

    return df


# ============================================================
# Graph statistics
# ============================================================

def graph_statistics(
    model,
    data: pd.DataFrame,
) -> dict:
    """
    Calculate structural complexity statistics.
    """

    edges = list(
        model.edges()
    )

    number_of_edges = len(edges)

    indegrees = {
        node: int(
            model.in_degree(node)
        )
        for node in model.nodes()
    }

    maximum_indegree = max(
        indegrees.values()
    )

    cpt_information = {}

    for node in model.nodes():

        parents = list(
            model.get_parents(node)
        )

        node_states = (
            data[node]
            .nunique()
        )

        parent_configurations = 1

        for parent in parents:

            parent_configurations *= (
                data[parent]
                .nunique()
            )

        cpt_entries = (
            parent_configurations
            * node_states
        )

        cpt_information[node] = {
            "parents": parents,
            "node_states": int(
                node_states
            ),
            "parent_configurations": int(
                parent_configurations
            ),
            "cpt_entries": int(
                cpt_entries
            ),
        }

    total_cpt_entries = sum(
        item["cpt_entries"]
        for item in cpt_information.values()
    )

    return {
        "number_of_edges": (
            number_of_edges
        ),
        "maximum_indegree": (
            maximum_indegree
        ),
        "total_cpt_entries": int(
            total_cpt_entries
        ),
        "cpt_information": (
            cpt_information
        ),
    }


# ============================================================
# Score model
# ============================================================

def calculate_scores(
    model,
    data: pd.DataFrame,
) -> dict:
    """
    Calculate BIC and BDeu scores for the learned graph.

    Higher scores are better.
    """

    bic_score = BIC(
        data
    )

    bdeu_score = BDeu(
        data,
        equivalent_sample_size=(
            BDEU_EQUIVALENT_SAMPLE_SIZE
        ),
    )

    return {
        "bic_score": float(
            bic_score.score(model)
        ),
        "bdeu_score": float(
            bdeu_score.score(model)
        ),
    }


# ============================================================
# Validate temporal/domain constraints
# ============================================================

def validate_constraints(
    model,
) -> None:
    """
    Verify that the learned graph does not contain
    temporally invalid edges.
    """

    baseline_variables = [
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

    # --------------------------------------------------------
    # Outcome must not point backwards to treatment
    # --------------------------------------------------------

    if model.has_edge(
        "label",
        "trt",
    ):
        raise ValueError(
            "Invalid learned graph: "
            "label -> trt"
        )

    # --------------------------------------------------------
    # Treatment must not point backwards to baseline
    # --------------------------------------------------------

    for variable in baseline_variables:

        if model.has_edge(
            "trt",
            variable,
        ):
            raise ValueError(
                f"Invalid learned graph: "
                f"trt -> {variable}"
            )

    # --------------------------------------------------------
    # Outcome must not point backwards to baseline
    # --------------------------------------------------------

    for variable in baseline_variables:

        if model.has_edge(
            "label",
            variable,
        ):
            raise ValueError(
                f"Invalid learned graph: "
                f"label -> {variable}"
            )


# ============================================================
# Learn constrained Hill Climbing graph
# ============================================================

def learn_hill_climbing(
    name: str,
    data: pd.DataFrame,
):
    """
    Learn a constrained Bayesian-network structure using
    Hill Climbing with BIC as the optimization score.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"CONSTRAINED HILL CLIMBING — "
        f"{name.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDataset shape: "
        f"{data.shape}"
    )

    print(
        f"Variables: "
        f"{len(VARIABLES)}"
    )

    print(
        f"Maximum indegree: "
        f"{MAX_INDEGREE}"
    )

    # --------------------------------------------------------
    # Expert knowledge
    # --------------------------------------------------------

    expert_knowledge = (
        build_expert_knowledge()
    )

    print(
        "\nExpert knowledge:"
    )

    print(
        expert_knowledge
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # pgmpy 1.1.2 places max_indegree in the constructor.
    # It must NOT be passed to fit().
    # --------------------------------------------------------

    search = HillClimbSearch(
        scoring_method="bic-d",
        max_indegree=MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        return_type="dag",
        max_iter=MAX_ITER,
        epsilon=EPSILON,
        show_progress=SHOW_PROGRESS,
    )

    # --------------------------------------------------------
    # Fit the structure
    # --------------------------------------------------------

    fitted_search = search.fit(
        data
    )

    model = (
        fitted_search
        .causal_graph_
    )

    # --------------------------------------------------------
    # Validate constraints
    # --------------------------------------------------------

    validate_constraints(
        model
    )

    print(
        "\nConstraint validation: PASSED"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = graph_statistics(
        model,
        data,
    )

    scores = calculate_scores(
        model,
        data,
    )

    # --------------------------------------------------------
    # Print edges
    # --------------------------------------------------------

    print(
        "\nLearned edges:"
    )

    for source, target in sorted(
        model.edges()
    ):
        print(
            f"  {source} -> {target}"
        )

    # --------------------------------------------------------
    # Graph statistics
    # --------------------------------------------------------

    print(
        "\nGraph statistics:"
    )

    print(
        f"  Number of edges: "
        f"{stats['number_of_edges']}"
    )

    print(
        f"  Maximum indegree: "
        f"{stats['maximum_indegree']}"
    )

    print(
        f"  Total CPT entries: "
        f"{stats['total_cpt_entries']}"
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    print(
        "\nScores:"
    )

    print(
        f"  BIC: "
        f"{scores['bic_score']:.4f}"
    )

    print(
        f"  BDeu: "
        f"{scores['bdeu_score']:.4f}"
    )

    return (
        model,
        stats,
        scores,
    )


# ============================================================
# Save graph edges
# ============================================================

def save_edges(
    name: str,
    model,
) -> Path:

    candidate_dir = (
        OUTPUT_DIR / name
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges_path = (
        candidate_dir
        / "hill_climbing_constrained_edges.csv"
    )

    edges_df = pd.DataFrame(
        list(model.edges()),
        columns=[
            "source",
            "target",
        ],
    )

    edges_df.to_csv(
        edges_path,
        index=False,
    )

    return edges_path


# ============================================================
# Save statistics
# ============================================================

def save_statistics(
    name: str,
    stats: dict,
    scores: dict,
) -> Path:

    candidate_dir = (
        OUTPUT_DIR / name
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics_path = (
        candidate_dir
        / "hill_climbing_constrained_statistics.json"
    )

    output = {
        "representation": name,
        "algorithm": (
            "Constrained Hill Climbing"
        ),
        "optimization_score": "BIC",
        "secondary_score": "BDeu",
        "max_indegree": (
            MAX_INDEGREE
        ),
        "bdeu_equivalent_sample_size": (
            BDEU_EQUIVALENT_SAMPLE_SIZE
        ),
        "max_iterations": MAX_ITER,
        "epsilon": EPSILON,
        "constraints": {
            "temporal_order": [
                "baseline",
                "treatment",
                "outcome",
            ],
            "forbidden_edges": [
                [
                    "label",
                    "trt",
                ]
            ],
        },
        "scores": scores,
        "graph_statistics": {
            key: value
            for key, value in stats.items()
            if key != "cpt_information"
        },
        "cpt_information": (
            stats[
                "cpt_information"
            ]
        ),
    }

    with open(
        statistics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    return statistics_path


# ============================================================
# Save all results for one candidate
# ============================================================

def save_results(
    name: str,
    model,
    stats: dict,
    scores: dict,
):

    edges_path = save_edges(
        name,
        model,
    )

    statistics_path = save_statistics(
        name,
        stats,
        scores,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {edges_path}"
    )

    print(
        f"  {statistics_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ACTG175 CONSTRAINED HILL CLIMBING"
    )

    print(
        "SPARSE vs RICH"
    )

    print(
        "=" * 70
    )

    summary = []

    # --------------------------------------------------------
    # Evaluate both representations
    # --------------------------------------------------------

    for name in CANDIDATES:

        data = load_candidate(
            name
        )

        (
            model,
            stats,
            scores,
        ) = learn_hill_climbing(
            name,
            data,
        )

        save_results(
            name,
            model,
            stats,
            scores,
        )

        summary.append(
            {
                "representation": name,
                "edges": stats[
                    "number_of_edges"
                ],
                "max_indegree": stats[
                    "maximum_indegree"
                ],
                "total_cpt_entries": stats[
                    "total_cpt_entries"
                ],
                "bic_score": scores[
                    "bic_score"
                ],
                "bdeu_score": scores[
                    "bdeu_score"
                ],
            }
        )

    # --------------------------------------------------------
    # Summary dataframe
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary
    )

    summary_path = (
        OUTPUT_DIR
        / "hill_climbing_constrained_comparison.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONSTRAINED HILL CLIMBING COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        summary_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary to:"
        f"\n{summary_path}"
    )

    # --------------------------------------------------------
    # Basic comparison
    # --------------------------------------------------------

    sparse = summary_df[
        summary_df["representation"]
        == "sparse"
    ].iloc[0]

    rich = summary_df[
        summary_df["representation"]
        == "rich"
    ].iloc[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRELIMINARY STRUCTURE COMPARISON"
    )

    print(
        "=" * 70
    )

    # BIC
    if (
        sparse["bic_score"]
        > rich["bic_score"]
    ):
        print(
            "BIC: Sparse is higher."
        )
    elif (
        rich["bic_score"]
        > sparse["bic_score"]
    ):
        print(
            "BIC: Rich is higher."
        )
    else:
        print(
            "BIC: Equal."
        )

    # BDeu
    if (
        sparse["bdeu_score"]
        > rich["bdeu_score"]
    ):
        print(
            "BDeu: Sparse is higher."
        )
    elif (
        rich["bdeu_score"]
        > sparse["bdeu_score"]
    ):
        print(
            "BDeu: Rich is higher."
        )
    else:
        print(
            "BDeu: Equal."
        )

    # CPT complexity
    if (
        sparse["total_cpt_entries"]
        < rich["total_cpt_entries"]
    ):
        print(
            "CPT complexity: Sparse is lower."
        )
    elif (
        rich["total_cpt_entries"]
        < sparse["total_cpt_entries"]
    ):
        print(
            "CPT complexity: Rich is lower."
        )
    else:
        print(
            "CPT complexity: Equal."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONSTRAINED HILL CLIMBING COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()