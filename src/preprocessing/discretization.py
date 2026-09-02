import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEVELOPMENT_PATH = (
    PROCESSED_DIR / "actg175_development.csv"
)

TEST_PATH = (
    PROCESSED_DIR / "actg175_test.csv"
)

SPARSE_DIR = PROCESSED_DIR / "sparse"
RICH_DIR = PROCESSED_DIR / "rich"


# ============================================================
# Variables
# ============================================================

# Variables that will be discretized.
DISCRETIZE_VARIABLES = [
    "age",
    "wtkg",
    "karnof",
    "cd40",
    "cd80",
]

# preanti receives special handling because zero has
# a meaningful interpretation as no prior antiretroviral
# exposure.
SPECIAL_VARIABLE = "preanti"

# These remain unchanged.
UNCHANGED_VARIABLES = [
    "hemo",
    "homo",
    "drugs",
    "oprior",
    "z30",
    "race",
    "gender",
    "strat",
    "symptom",
    "trt",
    "label",
]


# ============================================================
# Quantile edge calculation
# ============================================================

def calculate_quantile_edges(
    series: pd.Series,
    n_bins: int,
) -> list[float]:
    """
    Calculate quantile-based bin boundaries using development
    data only.

    Infinite outer boundaries ensure that test observations
    outside the development range can still be assigned to a bin.
    """

    quantiles = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    values = series.dropna()

    raw_edges = values.quantile(
        quantiles
    ).to_numpy()

    # Remove duplicate boundaries caused by tied values.
    unique_edges = np.unique(raw_edges)

    if len(unique_edges) < 2:
        raise ValueError(
            f"Unable to create bins for {series.name}. "
            f"Only one unique boundary was found."
        )

    # Open the outer boundaries.
    unique_edges[0] = -np.inf
    unique_edges[-1] = np.inf

    return unique_edges.tolist()


# ============================================================
# Apply ordinary quantile discretization
# ============================================================

def discretize_variable(
    series: pd.Series,
    edges: list[float],
    prefix: str,
) -> pd.Series:

    n_bins = len(edges) - 1

    labels = [
        f"{prefix}{i + 1}"
        for i in range(n_bins)
    ]

    return pd.cut(
        series,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )


# ============================================================
# Special handling for preanti
# ============================================================

def calculate_preanti_edges(
    series: pd.Series,
    n_positive_bins: int,
) -> list[float]:

    positive_values = series[
        series > 0
    ].dropna()

    if positive_values.empty:
        raise ValueError(
            "No positive preanti observations found."
        )

    quantiles = np.linspace(
        0,
        1,
        n_positive_bins + 1,
    )

    raw_edges = positive_values.quantile(
        quantiles
    ).to_numpy()

    unique_edges = np.unique(raw_edges)

    if len(unique_edges) < 2:
        raise ValueError(
            "Unable to create positive preanti bins."
        )

    unique_edges[0] = 0
    unique_edges[-1] = np.inf

    return unique_edges.tolist()


def discretize_preanti(
    series: pd.Series,
    edges: list[float],
) -> pd.Series:
    """
    Create:
        0 = no prior exposure

    and positive-exposure quantile bins.
    """

    positive_bins = len(edges) - 1

    labels = [
        f"positive_{i + 1}"
        for i in range(positive_bins)
    ]

    result = pd.Series(
        index=series.index,
        dtype="object",
    )

    # Explicit zero state.
    result.loc[series == 0] = "zero"

    # Positive values.
    positive_mask = series > 0

    positive_values = pd.cut(
        series.loc[positive_mask],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    result.loc[positive_mask] = (
        positive_values.astype(object)
    )

    return result


# ============================================================
# Build candidate representation
# ============================================================

def build_candidate(
    development: pd.DataFrame,
    test: pd.DataFrame,
    n_bins: int,
    name: str,
):
    """
    Fit discretization boundaries on development data only
    and apply them to both development and test data.
    """

    development_result = development.copy()
    test_result = test.copy()

    metadata = {
        "representation": name,
        "n_bins": n_bins,
        "fit_dataset": "development_only",
        "variables": {},
    }

    # --------------------------------------------------------
    # Standard continuous variables
    # --------------------------------------------------------

    for column in DISCRETIZE_VARIABLES:

        edges = calculate_quantile_edges(
            development[column],
            n_bins,
        )

        development_result[column] = (
            discretize_variable(
                development[column],
                edges,
                prefix=f"{column}_",
            )
        )

        test_result[column] = (
            discretize_variable(
                test[column],
                edges,
                prefix=f"{column}_",
            )
        )

        metadata["variables"][column] = {
            "method": "quantile",
            "bins": n_bins,
            "edges": edges,
        }

    # --------------------------------------------------------
    # preanti
    # --------------------------------------------------------

    preanti_edges = calculate_preanti_edges(
        development[SPECIAL_VARIABLE],
        n_bins,
    )

    development_result[SPECIAL_VARIABLE] = (
        discretize_preanti(
            development[SPECIAL_VARIABLE],
            preanti_edges,
        )
    )

    test_result[SPECIAL_VARIABLE] = (
        discretize_preanti(
            test[SPECIAL_VARIABLE],
            preanti_edges,
        )
    )

    metadata["variables"][SPECIAL_VARIABLE] = {
        "method": "zero_plus_positive_quantiles",
        "positive_bins": n_bins,
        "edges": preanti_edges,
    }

    return (
        development_result,
        test_result,
        metadata,
    )


# ============================================================
# State-support report
# ============================================================

def print_state_support(
    name: str,
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print(f"{name.upper()} STATE SUPPORT")
    print("=" * 70)

    variables = (
        DISCRETIZE_VARIABLES
        + [SPECIAL_VARIABLE]
    )

    for column in variables:

        print(f"\n{column}")

        dev_counts = (
            development[column]
            .value_counts(dropna=False)
            .sort_index()
        )

        test_counts = (
            test[column]
            .value_counts(dropna=False)
            .sort_index()
        )

        report = pd.DataFrame(
            {
                "development": dev_counts,
                "test": test_counts,
            }
        ).fillna(0).astype(int)

        print(report.to_string())


# ============================================================
# Save candidate
# ============================================================

def save_candidate(
    directory: Path,
    development: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict,
) -> None:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = (
        directory / "development.csv"
    )

    test_path = (
        directory / "test.csv"
    )

    metadata_path = (
        directory / "discretization_metadata.json"
    )

    development.to_csv(
        development_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
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

    print("\nSaved:")
    print(f"  {development_path}")
    print(f"  {test_path}")
    print(f"  {metadata_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-2 DISCRETIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    development = pd.read_csv(
        DEVELOPMENT_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    print(
        f"\nDevelopment shape: {development.shape}"
    )

    print(
        f"Test shape:        {test.shape}"
    )

    # --------------------------------------------------------
    # Sparse representation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING SPARSE REPRESENTATION")
    print("=" * 70)

    sparse_dev, sparse_test, sparse_metadata = (
        build_candidate(
            development,
            test,
            n_bins=3,
            name="sparse",
        )
    )

    print_state_support(
        "sparse",
        sparse_dev,
        sparse_test,
    )

    save_candidate(
        SPARSE_DIR,
        sparse_dev,
        sparse_test,
        sparse_metadata,
    )

    # --------------------------------------------------------
    # Rich representation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING RICH REPRESENTATION")
    print("=" * 70)

    rich_dev, rich_test, rich_metadata = (
        build_candidate(
            development,
            test,
            n_bins=4,
            name="rich",
        )
    )

    print_state_support(
        "rich",
        rich_dev,
        rich_test,
    )

    save_candidate(
        RICH_DIR,
        rich_dev,
        rich_test,
        rich_metadata,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE-2 CANDIDATE DISCRETIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()