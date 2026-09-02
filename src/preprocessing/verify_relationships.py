from pathlib import Path

import pandas as pd

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "AIDS_ClinicalTrial_GroupStudy175.csv"
)


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the raw ACTG175 dataset without modifying it."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# Verify str2 relationship with strat
# ============================================================

def verify_str2_strat(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("1. STR2 vs STRAT RELATIONSHIP")
    print("=" * 70)

    crosstab = pd.crosstab(
        df["strat"],
        df["str2"],
        margins=True
    )

    print("\nCross-tabulation:")
    print(crosstab.to_string())

    print("\nUnique str2 values for each strat value:")

    for value in sorted(df["strat"].unique()):
        subset = df.loc[df["strat"] == value, "str2"]
        unique_values = sorted(subset.unique())

        print(
            f"strat={value} -> str2 values = {unique_values}"
        )

    # Check whether every strat value maps to exactly one str2 value.
    mapping_counts = (
        df.groupby("strat")["str2"]
        .nunique()
    )

    deterministic = bool((mapping_counts == 1).all())

    print(
        f"\nIs str2 deterministically determined by strat? "
        f"{deterministic}"
    )


# ============================================================
# Verify treat relationship with trt
# ============================================================

def verify_treat_trt(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("2. TREAT vs TRT RELATIONSHIP")
    print("=" * 70)

    crosstab = pd.crosstab(
        df["trt"],
        df["treat"],
        margins=True
    )

    print("\nCross-tabulation:")
    print(crosstab.to_string())

    print("\nTreatment mapping:")

    for value in sorted(df["trt"].unique()):
        subset = df.loc[df["trt"] == value, "treat"]
        unique_values = sorted(subset.unique())

        print(
            f"trt={value} -> treat values = {unique_values}"
        )

    mapping_counts = (
        df.groupby("trt")["treat"]
        .nunique()
    )

    deterministic = bool((mapping_counts == 1).all())

    print(
        f"\nIs treat deterministically determined by trt? "
        f"{deterministic}"
    )


# ============================================================
# Treatment vs baseline variables
# ============================================================

def inspect_treatment_relationships(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("3. TREATMENT vs BASELINE VARIABLES")
    print("=" * 70)

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

    print("\nMean baseline values by treatment:")
    print(
        df.groupby("trt")[baseline_variables]
        .mean()
        .round(3)
        .to_string()
    )

    print("\nTreatment counts:")
    print(
        df["trt"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# Outcome vs baseline variables
# ============================================================

def inspect_outcome_relationships(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("4. OUTCOME vs BASELINE VARIABLES")
    print("=" * 70)

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

    print("\nMean baseline values by outcome:")
    print(
        df.groupby("label")[baseline_variables]
        .mean()
        .round(3)
        .to_string()
    )

    print("\nOutcome counts:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# Outcome by treatment
# ============================================================

def treatment_outcome_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("5. OUTCOME BY TREATMENT")
    print("=" * 70)

    counts = pd.crosstab(
        df["trt"],
        df["label"]
    )

    print("\nCounts:")
    print(counts.to_string())

    percentages = pd.crosstab(
        df["trt"],
        df["label"],
        normalize="index"
    ) * 100

    print("\nRow percentages:")
    print(percentages.round(2).to_string())


# ============================================================
# Post-treatment variables
# ============================================================

def inspect_post_treatment_variables(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("6. POST-TREATMENT VARIABLES")
    print("=" * 70)

    post_treatment = [
        "cd420",
        "cd820",
        "offtrt",
    ]

    print("\nMean post-treatment measurements by treatment:")

    available = [
        column
        for column in post_treatment
        if column in df.columns
    ]

    print(
        df.groupby("trt")[available]
        .mean()
        .round(3)
        .to_string()
    )

    print("\nPost-treatment variable distributions:")

    for column in post_treatment:
        print(f"\n{column}:")
        print(
            df[column]
            .value_counts()
            .sort_index()
            .head(20)
            .to_string()
        )


# ============================================================
# Continuous variable distributions
# ============================================================

def inspect_continuous_variables(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("7. CONTINUOUS VARIABLE DISTRIBUTIONS")
    print("=" * 70)

    variables = [
        "age",
        "wtkg",
        "preanti",
        "karnof",
        "cd40",
        "cd80",
    ]

    summary = df[variables].describe().T

    summary["skewness"] = df[variables].skew()

    print(
        summary[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "skewness",
            ]
        ]
        .round(3)
        .to_string()
    )


# ============================================================
# Correlation matrix
# ============================================================

def inspect_numeric_correlations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("8. NUMERIC CORRELATIONS")
    print("=" * 70)

    variables = [
        "age",
        "wtkg",
        "preanti",
        "karnof",
        "cd40",
        "cd80",
        "time",
        "label",
    ]

    correlation = pd.DataFrame(df[variables]).corr()

    print(
        correlation.round(3).to_string()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("ACTG175 RELATIONSHIP AND ENCODING VERIFICATION")
    print("=" * 70)

    df = load_dataset()

    verify_str2_strat(df)

    verify_treat_trt(df)

    inspect_treatment_relationships(df)

    inspect_outcome_relationships(df)

    treatment_outcome_table(df)

    inspect_post_treatment_variables(df)

    inspect_continuous_variables(df)

    inspect_numeric_correlations(df)

    print("\n" + "=" * 70)
    print("RELATIONSHIP VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()