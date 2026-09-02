import json
from pathlib import Path

import pandas as pd
from pgmpy.causal_discovery import GES
from pgmpy.structure_score import BIC, BDeu

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

SPARSE_DIR = (
    PROCESSED_DIR / "sparse"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "sparse"
    / "ges"
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


# ============================================================
# CONFIGURATION
# ============================================================

MIN_IMPROVEMENT = 1e-6

BDEU_EQUIVALENT_SAMPLE_SIZE = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

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
    # Schema
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
            f"Dataset contains {missing} missing values."
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
# GRAPH STATISTICS
# ============================================================

def calculate_graph_statistics(
    model,
    data: pd.DataFrame,
) -> dict:

    edges = list(
        model.edges()
    )

    number_of_edges = len(
        edges
    )

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
# SCORE CALCULATION
# ============================================================

def calculate_scores(
    model,
    data: pd.DataFrame,
) -> dict:

    bic = BIC(
        data
    )

    bdeu = BDeu(
        data,
        equivalent_sample_size=(
            BDEU_EQUIVALENT_SAMPLE_SIZE
        ),
    )

    return {
        "bic_score": float(
            bic.score(model)
        ),
        "bdeu_score": float(
            bdeu.score(model)
        ),
    }


# ============================================================
# TEMPORAL VIOLATION CHECK
# ============================================================

BASELINE_VARIABLES = {
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

TREATMENT_VARIABLES = {
    "trt",
}

OUTCOME_VARIABLES = {
    "label",
}


def temporal_tier(
    variable: str,
) -> int:

    if variable in BASELINE_VARIABLES:
        return 0

    if variable in TREATMENT_VARIABLES:
        return 1

    if variable in OUTCOME_VARIABLES:
        return 2

    raise ValueError(
        f"Unknown variable: {variable}"
    )


def is_temporally_valid(
    source: str,
    target: str,
) -> bool:

    source_tier = temporal_tier(
        source
    )

    target_tier = temporal_tier(
        target
    )

    # Same tier or earlier -> later is allowed; later -> earlier is not allowed.
    return source_tier <= target_tier


# ============================================================
# VALIDATE GES RESULT
# ============================================================

def find_temporal_violations(
    model,
) -> list:

    violations = []

    for source, target in model.edges():

        if not is_temporally_valid(
            source,
            target,
        ):
            violations.append(
                (
                    source,
                    target,
                )
            )

    return violations


# ============================================================
# RUN NATIVE GES
# ============================================================

def learn_ges(
    data: pd.DataFrame,
):

    print(
        "=" * 70
    )

    print(
        "ACTG175 NATIVE GES - SPARSE"
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
        f"Minimum improvement: "
        f"{MIN_IMPROVEMENT}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is NATIVE pgmpy GES."
    )

    print(
        "pgmpy 1.1.2 does not expose "
        "ExpertKnowledge for GES."
    )

    print(
        "Therefore temporal constraints are "
        "NOT imposed during GES."
    )

    print(
        "The resulting graph is used only "
        "for algorithmic sensitivity comparison."
    )

    # --------------------------------------------------------
    # GES
    # --------------------------------------------------------

    search = GES(
        scoring_method="bic-d",
        return_type="dag",
        min_improvement=(
            MIN_IMPROVEMENT
        ),
    )

    fitted = search.fit(
        data
    )

    model = (
        fitted.causal_graph_
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
            f"{source} -> {target}"
        )

    # --------------------------------------------------------
    # Statistics
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

    # --------------------------------------------------------
    # Temporal violations
    # --------------------------------------------------------

    violations = (
        find_temporal_violations(
            model
        )
    )

    print(
        "\nTemporal validation:"
    )

    print(
        f"Violating edges: "
        f"{len(violations)}"
    )

    if violations:

        for source, target in violations:

            print(
                f"  {source} -> {target}"
            )

    else:

        print(
            "No temporal violations."
        )

    return (
        model,
        statistics,
        scores,
        violations,
    )


# ============================================================
# SAVE EDGES
# ============================================================

def save_edges(
    model,
) -> Path:

    path = (
        OUTPUT_DIR
        / "ges_edges.csv"
    )

    edges_df = pd.DataFrame(
        [{"source": str(u), "target": str(v)} for u, v in model.edges()]
    )

    edges_df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    statistics: dict,
    scores: dict,
    violations: list,
) -> Path:

    path = (
        OUTPUT_DIR
        / "ges_statistics.json"
    )

    output = {
        "representation": "sparse",

        "algorithm": (
            "Native GES"
        ),

        "implementation": (
            "pgmpy 1.1.2 GES"
        ),

        "purpose": (
            "Algorithmic sensitivity comparison"
        ),

        "optimization_score": "BIC",

        "secondary_score": "BDeu",

        "min_improvement": (
            MIN_IMPROVEMENT
        ),

        "bdeu_equivalent_sample_size": (
            BDEU_EQUIVALENT_SAMPLE_SIZE
        ),

        "constraints": {
            "used_during_learning": False,

            "reason": (
                "pgmpy 1.1.2 GES does not "
                "expose ExpertKnowledge"
            ),

            "temporal_rule_checked_after_learning": (
                "baseline -> treatment -> outcome"
            ),
        },

        "scores": scores,

        "temporal_violations": [
            {
                "source": source,
                "target": target,
            }
            for source, target
            in violations
        ],

        "graph_statistics": {
            key: value
            for key, value
            in statistics.items()
            if key != "cpt_information"
        },

        "cpt_information": (
            statistics[
                "cpt_information"
            ]
        ),
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    return path


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_edges(
    path: Path,
) -> set:

    if not path.exists():
        return set()

    df = pd.read_csv(
        path
    )

    return set(
        zip(
            df["source"],
            df["target"],
        )
    )


def load_statistics(
    path: Path,
):

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# JACCARD
# ============================================================

def edge_jaccard(
    first: set,
    second: set,
) -> float:

    union = (
        first | second
    )

    if not union:
        return 1.0

    return (
        len(first & second)
        / len(union)
    )


# ============================================================
# COMPARE HC / TABU / GES
# ============================================================

def compare_algorithms(
    ges_model,
    ges_statistics,
    ges_scores,
):

    hc_edges_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "hill_climbing_constrained_edges.csv"
    )

    tabu_edges_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "tabu_search"
        / "tabu_search_edges.csv"
    )

    hc_stats_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "hill_climbing_constrained_statistics.json"
    )

    tabu_stats_path = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "sparse"
        / "tabu_search"
        / "tabu_search_statistics.json"
    )

    hc_edges = load_edges(
        hc_edges_path
    )

    tabu_edges = load_edges(
        tabu_edges_path
    )

    ges_edges = set(
        ges_model.edges()
    )

    hc_stats = load_statistics(
        hc_stats_path
    )

    tabu_stats = load_statistics(
        tabu_stats_path
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HC vs TABU vs NATIVE GES"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # BIC
    # --------------------------------------------------------

    print(
        "\nBIC:"
    )

    if hc_stats:

        print(
            f"Hill Climbing: "
            f"{hc_stats['scores']['bic_score']:.4f}"
        )

    if tabu_stats:

        print(
            f"Tabu Search:   "
            f"{tabu_stats['scores']['bic_score']:.4f}"
        )

    print(
        f"Native GES:    "
        f"{ges_scores['bic_score']:.4f}"
    )

    # --------------------------------------------------------
    # BDeu
    # --------------------------------------------------------

    print(
        "\nBDeu:"
    )

    if hc_stats:

        print(
            f"Hill Climbing: "
            f"{hc_stats['scores']['bdeu_score']:.4f}"
        )

    if tabu_stats:

        print(
            f"Tabu Search:   "
            f"{tabu_stats['scores']['bdeu_score']:.4f}"
        )

    print(
        f"Native GES:    "
        f"{ges_scores['bdeu_score']:.4f}"
    )

    # --------------------------------------------------------
    # Edge counts
    # --------------------------------------------------------

    print(
        "\nEdges:"
    )

    print(
        f"Hill Climbing: "
        f"{len(hc_edges)}"
    )

    print(
        f"Tabu Search:   "
        f"{len(tabu_edges)}"
    )

    print(
        f"Native GES:    "
        f"{len(ges_edges)}"
    )

    # --------------------------------------------------------
    # Jaccard
    # --------------------------------------------------------

    print(
        "\nEdge Jaccard:"
    )

    print(
        f"HC vs TABU: "
        f"{edge_jaccard(hc_edges, tabu_edges):.4f}"
    )

    print(
        f"HC vs GES:  "
        f"{edge_jaccard(hc_edges, ges_edges):.4f}"
    )

    print(
        f"TABU vs GES:"
        f" {edge_jaccard(tabu_edges, ges_edges):.4f}"
    )

    # --------------------------------------------------------
    # CPT
    # --------------------------------------------------------

    print(
        "\nCPT entries:"
    )

    if hc_stats:

        print(
            f"Hill Climbing: "
            f"{hc_stats['graph_statistics']['total_cpt_entries']}"
        )

    if tabu_stats:

        print(
            f"Tabu Search:   "
            f"{tabu_stats['graph_statistics']['total_cpt_entries']}"
        )

    print(
        f"Native GES:    "
        f"{ges_statistics['total_cpt_entries']}"
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    rows = []

    if hc_stats:

        rows.append(
            {
                "algorithm": (
                    "Constrained Hill Climbing"
                ),
                "bic_score": (
                    hc_stats[
                        "scores"
                    ][
                        "bic_score"
                    ]
                ),
                "bdeu_score": (
                    hc_stats[
                        "scores"
                    ][
                        "bdeu_score"
                    ]
                ),
                "edges": len(
                    hc_edges
                ),
                "total_cpt_entries": (
                    hc_stats[
                        "graph_statistics"
                    ][
                        "total_cpt_entries"
                    ]
                ),
                "constraints_used": True,
            }
        )

    if tabu_stats:

        rows.append(
            {
                "algorithm": (
                    "Tabu Search"
                ),
                "bic_score": (
                    tabu_stats[
                        "scores"
                    ][
                        "bic_score"
                    ]
                ),
                "bdeu_score": (
                    tabu_stats[
                        "scores"
                    ][
                        "bdeu_score"
                    ]
                ),
                "edges": len(
                    tabu_edges
                ),
                "total_cpt_entries": (
                    tabu_stats[
                        "graph_statistics"
                    ][
                        "total_cpt_entries"
                    ]
                ),
                "constraints_used": True,
            }
        )

    rows.append(
        {
            "algorithm": "Native GES",
            "bic_score": (
                ges_scores[
                    "bic_score"
                ]
            ),
            "bdeu_score": (
                ges_scores[
                    "bdeu_score"
                ]
            ),
            "edges": len(
                ges_edges
            ),
            "total_cpt_entries": (
                ges_statistics[
                    "total_cpt_entries"
                ]
            ),
            "constraints_used": False,
        }
    )

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "ges_algorithm_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        f"\nSaved comparison:"
        f"\n{comparison_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ACTG175 STRUCTURE LEARNING"
    )

    print(
        "NATIVE GES - SPARSE"
    )

    print(
        "=" * 70
    )

    data = load_data()

    (
        model,
        statistics,
        scores,
        violations,
    ) = learn_ges(
        data
    )

    edges_path = save_edges(
        model
    )

    statistics_path = save_statistics(
        statistics,
        scores,
        violations,
    )

    print(
        "\nSaved:"
    )

    print(
        f"{edges_path}"
    )

    print(
        f"{statistics_path}"
    )

    compare_algorithms(
        model,
        statistics,
        scores,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NATIVE GES COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()