import json
from pathlib import Path

import networkx as nx
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = PROJECT_ROOT / "results" / "structure_learning"
SPARSE_ROOT = RESULTS_ROOT / "sparse"

# ------------------------------------------------------------
# Known paths
# ------------------------------------------------------------

HC_CONSTRAINED_PATH = (
    SPARSE_ROOT / "hill_climbing_constrained_edges.csv"
)

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

BOOTSTRAP_STABLE_PATH = (
    SPARSE_ROOT
    / "bootstrap"
    / "stable_edges.csv"
)

BOOTSTRAP_VERY_STABLE_PATH = (
    SPARSE_ROOT
    / "bootstrap"
    / "very_stable_edges.csv"
)

FINAL_DIR = RESULTS_ROOT / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


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
# AUTOMATIC TABU FILE DISCOVERY
# ============================================================

def find_tabu_edges_file():
    """
    Automatically locate the Tabu Search edge file.

    This avoids relying on a hard-coded nested directory because
    the project has used slightly different output paths during
    development.
    """

    if not SPARSE_ROOT.exists():
        raise FileNotFoundError(
            f"Sparse results directory not found:\n{SPARSE_ROOT}"
        )

    candidates = []

    # Search recursively for CSV files containing both
    # 'tabu' and 'edge' in the filename/path.
    for path in SPARSE_ROOT.rglob("*.csv"):

        text = str(path).lower()

        if "tabu" in text and "edge" in text:
            candidates.append(path)

    # Remove duplicates and sort
    candidates = sorted(
        set(candidates),
        key=lambda p: str(p).lower()
    )

    if not candidates:

        raise FileNotFoundError(
            "Could not automatically find the Tabu edge file.\n\n"
            f"Searched inside:\n{SPARSE_ROOT}\n\n"
            "CSV files containing 'tabu' and 'edge' were not found."
        )

    # Prefer files whose filename explicitly contains
    # tabu_search_edges.
    preferred = [
        p for p in candidates
        if "tabu_search_edges" in p.name.lower()
    ]

    if preferred:
        selected = preferred[0]
    else:
        selected = candidates[0]

    print("\nTabu edge file discovered automatically:")
    print(selected)

    if len(candidates) > 1:
        print("\nOther Tabu-related CSV files found:")
        for candidate in candidates:
            print(f"  {candidate}")

    return selected


# ============================================================
# LOAD EDGE FILE
# ============================================================

def load_edges(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    if {
        "source",
        "target",
    }.issubset(df.columns):

        source_column = "source"
        target_column = "target"

    elif {
        "from",
        "to",
    }.issubset(df.columns):

        source_column = "from"
        target_column = "to"

    else:

        raise ValueError(
            f"Cannot identify source/target columns in:\n"
            f"{path}\n\n"
            f"Columns found:\n"
            f"{df.columns.tolist()}"
        )

    edges = set()

    for _, row in df.iterrows():

        source = str(
            row[source_column]
        ).strip()

        target = str(
            row[target_column]
        ).strip()

        edges.add(
            (
                source,
                target,
            )
        )

    return edges


# ============================================================
# LOAD BOOTSTRAP
# ============================================================

def load_bootstrap_stability():

    if not BOOTSTRAP_STABLE_PATH.exists():

        raise FileNotFoundError(
            f"Bootstrap stability file not found:\n"
            f"{BOOTSTRAP_STABLE_PATH}"
        )

    df = pd.read_csv(
        BOOTSTRAP_STABLE_PATH
    )

    required = {
        "source",
        "target",
        "count",
        "stability",
        "stability_percentage",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "Bootstrap stable_edges.csv is missing:\n"
            + "\n".join(sorted(missing))
        )

    return df


# ============================================================
# DAG VALIDATION
# ============================================================

def validate_dag(edges):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        VARIABLES
    )

    graph.add_edges_from(
        edges
    )

    # --------------------------------------------------------
    # Unknown variables
    # --------------------------------------------------------

    unknown = (
        set(graph.nodes())
        - set(VARIABLES)
    )

    if unknown:

        raise ValueError(
            "Unknown variables found:\n"
            + "\n".join(sorted(unknown))
        )

    # --------------------------------------------------------
    # DAG check
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(graph):

        raise ValueError(
            "Final graph contains a directed cycle."
        )

    # --------------------------------------------------------
    # Temporal tiers
    # --------------------------------------------------------

    tier = {}

    for variable in BASELINE_VARIABLES:
        tier[variable] = 0

    for variable in TREATMENT_VARIABLES:
        tier[variable] = 1

    for variable in OUTCOME_VARIABLES:
        tier[variable] = 2

    violations = []

    for source, target in edges:

        if tier[source] > tier[target]:

            violations.append(
                (source, target)
            )

    # Explicit forbidden edge
    if ("label", "trt") in edges and ("label", "trt") not in violations:

        violations.append(
            ("label", "trt")
        )

    return violations


# ============================================================
# GRAPH COMPLEXITY
# ============================================================

def calculate_complexity(edges):

    if not DEVELOPMENT_PATH.exists():

        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DEVELOPMENT_PATH}"
        )

    df = pd.read_csv(
        DEVELOPMENT_PATH
    )

    graph = nx.DiGraph()

    graph.add_nodes_from(
        VARIABLES
    )

    graph.add_edges_from(
        edges
    )

    indegrees = dict(
        graph.in_degree()
    )

    maximum_indegree = max(
        indegrees.values()
    )

    total_cpt_entries = 0

    for node in VARIABLES:

        node_states = int(
            df[node].nunique()
        )

        parents = list(
            graph.predecessors(node)
        )

        parent_configurations = 1

        for parent in parents:

            parent_configurations *= int(
                df[parent].nunique()
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
# COMPARE HC AND TABU
# ============================================================

def compare_graphs(hc_edges, tabu_edges):

    common = (
        hc_edges & tabu_edges
    )

    union = (
        hc_edges | tabu_edges
    )

    if union:

        jaccard = (
            len(common) / len(union)
        )

    else:

        jaccard = 1.0

    return {
        "common": common,
        "hc_only": hc_edges - tabu_edges,
        "tabu_only": tabu_edges - hc_edges,
        "jaccard": jaccard,
    }


# ============================================================
# BOOTSTRAP SUPPORT
# ============================================================

def evaluate_bootstrap_support(
    final_edges,
    bootstrap_df,
):

    lookup = {}

    for _, row in bootstrap_df.iterrows():

        key = (
            str(row["source"]).strip(),
            str(row["target"]).strip(),
        )

        lookup[key] = float(
            row["stability"]
        )

    records = []

    for source, target in sorted(
        final_edges
    ):

        stability = lookup.get(
            (
                source,
                target,
            ),
            0.0,
        )

        reverse = lookup.get(
            (
                target,
                source,
            ),
            0.0,
        )

        if stability >= 0.75:

            category = "VERY_STABLE"

        elif stability >= 0.50:

            category = "STABLE"

        elif stability >= 0.25:

            category = "MODERATE"

        else:

            category = "WEAK"

        records.append(
            {
                "source": source,
                "target": target,
                "bootstrap_stability": stability,
                "bootstrap_percentage": (
                    stability * 100
                ),
                "reverse_stability": reverse,
                "reverse_percentage": (
                    reverse * 100
                ),
                "support_category": category,
            }
        )

    result = pd.DataFrame(
        records
    )

    if not result.empty:

        result = (
            result
            .sort_values(
                "bootstrap_stability",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return result, lookup


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

def direction_analysis(
    final_edges,
    bootstrap_lookup,
):

    records = []

    checked = set()

    for source, target in final_edges:

        pair = frozenset(
            [
                source,
                target,
            ]
        )

        if pair in checked:
            continue

        checked.add(pair)

        forward = bootstrap_lookup.get(
            (
                source,
                target,
            ),
            0.0,
        )

        reverse = bootstrap_lookup.get(
            (
                target,
                source,
            ),
            0.0,
        )

        if forward > 0 and reverse > 0:

            if forward >= reverse:

                preferred = (
                    f"{source} -> {target}"
                )

            else:

                preferred = (
                    f"{target} -> {source}"
                )

            records.append(
                {
                    "hc_direction": (
                        f"{source} -> {target}"
                    ),
                    "hc_stability": (
                        forward * 100
                    ),
                    "reverse_direction": (
                        f"{target} -> {source}"
                    ),
                    "reverse_stability": (
                        reverse * 100
                    ),
                    "preferred_bootstrap_direction": (
                        preferred
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "hc_direction",
            "hc_stability",
            "reverse_direction",
            "reverse_stability",
            "preferred_bootstrap_direction",
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 FINAL DAG SELECTION")
    print("=" * 70)

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    print("\nChecking required files...")

    if not HC_CONSTRAINED_PATH.exists():

        raise FileNotFoundError(
            f"HC constrained file missing:\n"
            f"{HC_CONSTRAINED_PATH}"
        )

    print(
        f"FOUND HC:\n"
        f"{HC_CONSTRAINED_PATH}"
    )

    # --------------------------------------------------------
    # Automatically find Tabu file
    # --------------------------------------------------------

    tabu_path = find_tabu_edges_file()

    if not BOOTSTRAP_STABLE_PATH.exists():

        raise FileNotFoundError(
            f"Bootstrap stable file missing:\n"
            f"{BOOTSTRAP_STABLE_PATH}"
        )

    print(
        f"\nFOUND Bootstrap:\n"
        f"{BOOTSTRAP_STABLE_PATH}"
    )

    if not DEVELOPMENT_PATH.exists():

        raise FileNotFoundError(
            f"Development data missing:\n"
            f"{DEVELOPMENT_PATH}"
        )

    print(
        f"\nFOUND Development data:\n"
        f"{DEVELOPMENT_PATH}"
    )

    # ========================================================
    # LOAD HC
    # ========================================================

    print("\n" + "=" * 70)
    print("CONSTRAINED HILL CLIMBING")
    print("=" * 70)

    hc_edges = load_edges(
        HC_CONSTRAINED_PATH
    )

    print(
        f"Edges: {len(hc_edges)}"
    )

    # ========================================================
    # LOAD TABU
    # ========================================================

    print("\n" + "=" * 70)
    print("CONSTRAINED TABU SEARCH")
    print("=" * 70)

    tabu_edges = load_edges(
        tabu_path
    )

    print(
        f"Edges: {len(tabu_edges)}"
    )

    # ========================================================
    # COMPARE
    # ========================================================

    comparison = compare_graphs(
        hc_edges,
        tabu_edges,
    )

    common = comparison["common"]
    hc_only = comparison["hc_only"]
    tabu_only = comparison["tabu_only"]
    jaccard = comparison["jaccard"]

    print("\n" + "=" * 70)
    print("HC vs TABU")
    print("=" * 70)

    print(
        f"HC edges:      {len(hc_edges)}"
    )

    print(
        f"Tabu edges:    {len(tabu_edges)}"
    )

    print(
        f"Common edges:  {len(common)}"
    )

    print(
        f"HC-only edges: {len(hc_only)}"
    )

    print(
        f"Tabu-only:     {len(tabu_only)}"
    )

    print(
        f"Jaccard:       {jaccard:.4f}"
    )

    if hc_only:

        print("\nHC-only edges:")

        for source, target in sorted(
            hc_only
        ):

            print(
                f"  {source} -> {target}"
            )

    if tabu_only:

        print("\nTabu-only edges:")

        for source, target in sorted(
            tabu_only
        ):

            print(
                f"  {source} -> {target}"
            )

    # ========================================================
    # LOAD BOOTSTRAP
    # ========================================================

    print("\n" + "=" * 70)
    print("BOOTSTRAP STABILITY")
    print("=" * 70)

    bootstrap_df = (
        load_bootstrap_stability()
    )

    print(
        f"Bootstrap records: "
        f"{len(bootstrap_df)}"
    )

    # ========================================================
    # FINAL DAG
    # ========================================================

    # Constrained HC is the primary candidate.
    #
    # HC and Tabu were already shown in the previous phase
    # to produce the same constrained graph.
    #
    # Bootstrap is used as robustness evidence.
    #
    # We do NOT automatically rewrite the DAG based only
    # on bootstrap frequencies.

    final_edges = set(
        hc_edges
    )

    print("\n" + "=" * 70)
    print("FINAL DAG CANDIDATE")
    print("=" * 70)

    print(
        "Selection method:"
    )

    print(
        "Constrained Hill Climbing"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    violations = validate_dag(
        final_edges
    )

    if violations:

        print(
            "\nFINAL DAG VALIDATION: FAILED"
        )

        for source, target in violations:

            print(
                f"Violation: "
                f"{source} -> {target}"
            )

        raise RuntimeError(
            "Final DAG violates constraints."
        )

    print(
        "\nFinal DAG validation: PASSED"
    )

    print(
        "DAG structure: PASSED"
    )

    print(
        "Temporal constraints: PASSED"
    )

    print(
        "Forbidden label -> trt: PASSED"
    )

    # ========================================================
    # BOOTSTRAP SUPPORT
    # ========================================================

    support_df, bootstrap_lookup = (
        evaluate_bootstrap_support(
            final_edges,
            bootstrap_df,
        )
    )

    very_stable = int(
        (
            support_df[
                "bootstrap_stability"
            ] >= 0.75
        ).sum()
    )

    stable = int(
        (
            support_df[
                "bootstrap_stability"
            ] >= 0.50
        ).sum()
    )

    moderate = int(
        (
            (
                support_df[
                    "bootstrap_stability"
                ] >= 0.25
            )
            &
            (
                support_df[
                    "bootstrap_stability"
                ] < 0.50
            )
        ).sum()
    )

    weak = int(
        (
            support_df[
                "bootstrap_stability"
            ] < 0.25
        ).sum()
    )

    print("\n" + "=" * 70)
    print("BOOTSTRAP SUPPORT")
    print("=" * 70)

    print(
        f"Very stable >=75%: "
        f"{very_stable}"
    )

    print(
        f"Stable >=50%:      "
        f"{stable}"
    )

    print(
        f"Moderate 25-49%:   "
        f"{moderate}"
    )

    print(
        f"Weak <25%:         "
        f"{weak}"
    )

    print(
        "\nFinal DAG edge support:"
    )

    for _, row in support_df.iterrows():

        print(
            f"{row['source']} -> "
            f"{row['target']} "
            f"["
            f"{row['bootstrap_percentage']:.1f}%"
            f"] "
            f"{row['support_category']}"
        )

    # ========================================================
    # DIRECTION ANALYSIS
    # ========================================================

    direction_df = direction_analysis(
        final_edges,
        bootstrap_lookup,
    )

    # ========================================================
    # COMPLEXITY
    # ========================================================

    complexity = calculate_complexity(
        final_edges
    )

    print("\n" + "=" * 70)
    print("FINAL GRAPH COMPLEXITY")
    print("=" * 70)

    print(
        f"Edges: "
        f"{complexity['edges']}"
    )

    print(
        f"Maximum indegree: "
        f"{complexity['maximum_indegree']}"
    )

    print(
        f"Total CPT entries: "
        f"{complexity['total_cpt_entries']}"
    )

    # ========================================================
    # PRINT FINAL EDGES
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DAG EDGES")
    print("=" * 70)

    for source, target in sorted(
        final_edges
    ):

        stability = bootstrap_lookup.get(
            (
                source,
                target,
            ),
            0.0,
        )

        print(
            f"{source} -> {target} "
            f"[bootstrap "
            f"{stability * 100:.1f}%]"
        )

    # ========================================================
    # SAVE FINAL EDGES
    # ========================================================

    final_edges_df = pd.DataFrame(
        [{"source": str(u), "target": str(v)} for u, v in sorted(final_edges)]
    )

    final_edges_path = (
        FINAL_DIR
        / "final_dag_edges.csv"
    )

    final_edges_df.to_csv(
        final_edges_path,
        index=False,
    )

    # ========================================================
    # SAVE SUPPORT
    # ========================================================

    support_path = (
        FINAL_DIR
        / "final_dag_edge_support.csv"
    )

    support_df.to_csv(
        support_path,
        index=False,
    )

    # ========================================================
    # SAVE DIRECTION ANALYSIS
    # ========================================================

    direction_path = (
        FINAL_DIR
        / "bootstrap_direction_analysis.csv"
    )

    direction_df.to_csv(
        direction_path,
        index=False,
    )

    # ========================================================
    # SAVE ALGORITHM COMPARISON
    # ========================================================

    algorithm_df = pd.DataFrame(
        [
            {
                "algorithm":
                    "Constrained Hill Climbing",
                "edges":
                    len(hc_edges),
                "selected":
                    True,
            },
            {
                "algorithm":
                    "Constrained Tabu Search",
                "edges":
                    len(tabu_edges),
                "selected":
                    False,
            },
        ]
    )

    algorithm_path = (
        FINAL_DIR
        / "algorithm_comparison.csv"
    )

    algorithm_df.to_csv(
        algorithm_path,
        index=False,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "dataset": "ACTG175",

        "representation": "sparse",

        "selection_method":
            "Constrained Hill Climbing",

        "algorithm_agreement": {

            "hill_climbing_edges":
                len(hc_edges),

            "tabu_edges":
                len(tabu_edges),

            "common_edges":
                len(common),

            "hc_only_edges":
                len(hc_only),

            "tabu_only_edges":
                len(tabu_only),

            "jaccard":
                jaccard,
        },

        "bootstrap_support": {

            "edges_ge_75_percent":
                very_stable,

            "edges_ge_50_percent":
                stable,

            "edges_25_to_49_percent":
                moderate,

            "edges_lt_25_percent":
                weak,
        },

        "final_graph": {

            "edges":
                complexity["edges"],

            "maximum_indegree":
                complexity[
                    "maximum_indegree"
                ],

            "total_cpt_entries":
                complexity[
                    "total_cpt_entries"
                ],
        },

        "constraints": {

            "temporal_order":
                "Baseline -> Treatment -> Outcome",

            "forbidden_edge":
                "label -> trt",
        },

        "validation": {

            "is_dag":
                True,

            "temporal_constraints":
                True,

            "forbidden_edges":
                True,
        },

        "status":
            "FINAL DAG SELECTED AND VALIDATED",
    }

    summary_path = (
        FINAL_DIR
        / "final_dag_selection.json"
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
    # OUTPUT FILES
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVED FINAL DAG RESULTS")
    print("=" * 70)

    print(
        final_edges_path
    )

    print(
        support_path
    )

    print(
        direction_path
    )

    print(
        algorithm_path
    )

    print(
        summary_path
    )

    print("\n" + "=" * 70)
    print("FINAL DAG SELECTION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()