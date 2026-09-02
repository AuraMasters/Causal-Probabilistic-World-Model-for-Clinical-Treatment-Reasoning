import json
import sys
from pathlib import Path

import pandas as pd
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.structure_score import BIC, BDeu

# ============================================================
# Local module import
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

# IMPORTANT:
# Sparse is our provisional selected representation.
#
# Rich was retained for the earlier representation comparison,
# but the main structure-learning pipeline now proceeds with
# Sparse.
#
SPARSE_DIR = (
    PROCESSED_DIR / "sparse"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "sparse"
    / "tabu_search"
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

TABU_LENGTH = 100

MAX_ITER = 1_000_000

EPSILON = 1e-4

BDEU_EQUIVALENT_SAMPLE_SIZE = 10

SHOW_PROGRESS = True


# ============================================================
# Load data
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the Sparse development dataset.

    The test set is intentionally not used during structure
    learning.
    """

    path = (
        SPARSE_DIR
        / "development.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Sparse development dataset not found:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

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

    # --------------------------------------------------------
    # Convert states to strings
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

def calculate_graph_statistics(
    model,
    data: pd.DataFrame,
) -> dict:
    """
    Calculate structural complexity and CPT burden.
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

        node_states = int(
            data[node].nunique()
        )

        parent_configurations = 1

        for parent in parents:

            parent_configurations *= int(
                data[parent].nunique()
            )

        cpt_entries = (
            parent_configurations
            * node_states
        )

        cpt_information[node] = {
            "parents": parents,
            "node_states": node_states,
            "parent_configurations": (
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
# Score calculation
# ============================================================

def calculate_scores(
    model,
    data: pd.DataFrame,
) -> dict:
    """
    Calculate BIC and BDeu scores.

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
# Constraint validation
# ============================================================

def validate_constraints(
    model,
) -> None:
    """
    Verify that no temporally invalid edge has been learned.
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
    # Outcome -> treatment is forbidden
    # --------------------------------------------------------

    if model.has_edge(
        "label",
        "trt",
    ):
        raise ValueError(
            "Invalid graph: "
            "label -> trt"
        )

    # --------------------------------------------------------
    # Treatment -> baseline is forbidden
    # --------------------------------------------------------

    for variable in baseline_variables:

        if model.has_edge(
            "trt",
            variable,
        ):
            raise ValueError(
                f"Invalid graph: "
                f"trt -> {variable}"
            )

    # --------------------------------------------------------
    # Outcome -> baseline is forbidden
    # --------------------------------------------------------

    for variable in baseline_variables:

        if model.has_edge(
            "label",
            variable,
        ):
            raise ValueError(
                f"Invalid graph: "
                f"label -> {variable}"
            )


# ============================================================
# Run Tabu Search
# ============================================================

def learn_tabu(
    data: pd.DataFrame,
):
    """
    Run Tabu Search using pgmpy's HillClimbSearch with a
    nonzero tabu list.

    pgmpy 1.1.2 implements the tabu mechanism inside
    HillClimbSearch rather than exposing a separate
    TabuSearch class.
    """

    print(
        "=" * 70
    )

    print(
        "ACTG175 TABU SEARCH - SPARSE"
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

    print(
        f"Tabu length: "
        f"{TABU_LENGTH}"
    )

    print(
        f"Maximum iterations: "
        f"{MAX_ITER}"
    )

    print(
        f"BDeu equivalent sample size: "
        f"{BDEU_EQUIVALENT_SAMPLE_SIZE}"
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
    # pgmpy 1.1.2 uses HillClimbSearch with tabu_length
    # for the tabu-search behavior.
    #
    # max_indegree belongs in the constructor.
    # --------------------------------------------------------

    search = HillClimbSearch(
        scoring_method="bic-d",
        tabu_length=TABU_LENGTH,
        max_indegree=MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        return_type="dag",
        max_iter=MAX_ITER,
        epsilon=EPSILON,
        show_progress=SHOW_PROGRESS,
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fitted_search = search.fit(
        data
    )

    model = (
        fitted_search
        .causal_graph_
    )

    # --------------------------------------------------------
    # Validate
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

    statistics = (
        calculate_graph_statistics(
            model,
            data,
        )
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
        f"Number of edges: "
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

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    print(
        "\nScores:"
    )

    print(
        f"BIC: "
        f"{scores['bic_score']:.4f}"
    )

    print(
        f"BDeu: "
        f"{scores['bdeu_score']:.4f}"
    )

    return (
        model,
        statistics,
        scores,
    )


# ============================================================
# Save edges
# ============================================================

def save_edges(
    model,
) -> Path:

    edges_path = (
        OUTPUT_DIR
        / "tabu_search_edges.csv"
    )

    edges_df = pd.DataFrame(
        [{"source": str(u), "target": str(v)} for u, v in model.edges()]
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
    statistics: dict,
    scores: dict,
) -> Path:

    statistics_path = (
        OUTPUT_DIR
        / "tabu_search_statistics.json"
    )

    output = {
        "representation": "sparse",
        "algorithm": "Tabu Search",
        "implementation": (
            "pgmpy HillClimbSearch "
            "with tabu_length"
        ),
        "optimization_score": "BIC",
        "secondary_score": "BDeu",
        "max_indegree": (
            MAX_INDEGREE
        ),
        "tabu_length": (
            TABU_LENGTH
        ),
        "max_iterations": (
            MAX_ITER
        ),
        "epsilon": EPSILON,
        "bdeu_equivalent_sample_size": (
            BDEU_EQUIVALENT_SAMPLE_SIZE
        ),
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
            for key, value in statistics.items()
            if key != "cpt_information"
        },
        "cpt_information": (
            statistics[
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
# Compare with constrained Hill Climbing
# ============================================================

def compare_with_hill_climbing(
    tabu_model,
    tabu_scores: dict,
    tabu_statistics: dict,
):
    """
    Compare the Tabu result with the constrained Hill
    Climbing result already saved on disk.
    """

    hc_statistics_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "hill_climbing_constrained_statistics.json"
    )

    hc_edges_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "hill_climbing_constrained_edges.csv"
    )

    if not hc_statistics_path.exists():
        print(
            "\nHill Climbing statistics file not found."
        )
        return

    if not hc_edges_path.exists():
        print(
            "\nHill Climbing edges file not found."
        )
        return

    with open(
        hc_statistics_path,
        "r",
        encoding="utf-8",
    ) as file:

        hc_data = json.load(file)

    hc_edges_df = pd.read_csv(
        hc_edges_path
    )

    hc_edges = set(
        zip(
            hc_edges_df["source"],
            hc_edges_df["target"],
        )
    )

    tabu_edges = set(
        tabu_model.edges()
    )

    common_edges = (
        hc_edges
        & tabu_edges
    )

    union_edges = (
        hc_edges
        | tabu_edges
    )

    if union_edges:
        edge_jaccard = (
            len(common_edges)
            / len(union_edges)
        )
    else:
        edge_jaccard = 1.0

    hc_bic = float(
        hc_data[
            "scores"
        ][
            "bic_score"
        ]
    )

    hc_bdeu = float(
        hc_data[
            "scores"
        ][
            "bdeu_score"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TABU vs CONSTRAINED HILL CLIMBING"
    )

    print(
        "=" * 70
    )

    print(
        "\nBIC:"
    )

    print(
        f"  Hill Climbing: "
        f"{hc_bic:.4f}"
    )

    print(
        f"  Tabu Search:   "
        f"{tabu_scores['bic_score']:.4f}"
    )

    print(
        "\nBDeu:"
    )

    print(
        f"  Hill Climbing: "
        f"{hc_bdeu:.4f}"
    )

    print(
        f"  Tabu Search:   "
        f"{tabu_scores['bdeu_score']:.4f}"
    )

    print(
        "\nEdges:"
    )

    print(
        f"  Hill Climbing: "
        f"{len(hc_edges)}"
    )

    print(
        f"  Tabu Search:   "
        f"{len(tabu_edges)}"
    )

    print(
        f"  Common edges:  "
        f"{len(common_edges)}"
    )

    print(
        f"  Edge Jaccard:  "
        f"{edge_jaccard:.4f}"
    )

    print(
        "\nCPT entries:"
    )

    print(
        f"  Hill Climbing: "
        f"{hc_data['graph_statistics']['total_cpt_entries']}"
    )

    print(
        f"  Tabu Search:   "
        f"{tabu_statistics['total_cpt_entries']}"
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "algorithm": (
                    "Constrained Hill Climbing"
                ),
                "bic_score": hc_bic,
                "bdeu_score": hc_bdeu,
                "edges": len(hc_edges),
                "common_edges": None,
                "edge_jaccard": None,
                "total_cpt_entries": (
                    hc_data[
                        "graph_statistics"
                    ][
                        "total_cpt_entries"
                    ]
                ),
            },
            {
                "algorithm": "Tabu Search",
                "bic_score": (
                    tabu_scores[
                        "bic_score"
                    ]
                ),
                "bdeu_score": (
                    tabu_scores[
                        "bdeu_score"
                    ]
                ),
                "edges": len(tabu_edges),
                "common_edges": (
                    len(common_edges)
                ),
                "edge_jaccard": (
                    edge_jaccard
                ),
                "total_cpt_entries": (
                    tabu_statistics[
                        "total_cpt_entries"
                    ]
                ),
            },
        ]
    )

    comparison_path = (
        OUTPUT_DIR
        / "tabu_vs_hill_climbing.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        f"\nSaved comparison:"
        f"\n{comparison_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ACTG175 TABU SEARCH"
    )

    print(
        "SPARSE REPRESENTATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nNOTE:"
    )

    print(
        "pgmpy 1.1.2 implements the tabu mechanism "
        "inside HillClimbSearch."
    )

    print(
        "A nonzero tabu_length is therefore used "
        "as the Tabu Search configuration."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Learn
    # --------------------------------------------------------

    (
        model,
        statistics,
        scores,
    ) = learn_tabu(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    edges_path = save_edges(
        model
    )

    statistics_path = save_statistics(
        statistics,
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

    # --------------------------------------------------------
    # Compare with HC
    # --------------------------------------------------------

    compare_with_hill_climbing(
        model,
        scores,
        statistics,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TABU SEARCH COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()