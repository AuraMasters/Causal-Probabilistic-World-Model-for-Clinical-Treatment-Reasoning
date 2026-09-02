from pathlib import Path

import pandas as pd

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AIDS_ClinicalTrial_GroupStudy175.csv"


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the raw ACTG175 dataset without modifying it."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    return df


# ============================================================
# Dataset overview
# ============================================================

def print_dataset_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Rows         : {df.shape[0]}")
    print(f"Columns      : {df.shape[1]}")

    print("\nColumn names:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")


# ============================================================
# Data types
# ============================================================

def print_data_types(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes.to_string())


# ============================================================
# Missing values
# ============================================================

def print_missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100

    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": missing_percentage.round(2),
        }
    )

    print(missing_report.to_string())

    total_missing = int(missing.sum())

    print(f"\nTotal missing values: {total_missing}")


# ============================================================
# Duplicate rows
# ============================================================

def print_duplicates(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DUPLICATE ROWS")
    print("=" * 70)

    duplicate_count = int(df.duplicated().sum())

    print(f"Duplicate rows: {duplicate_count}")


# ============================================================
# Constant variables
# ============================================================

def print_constant_columns(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CONSTANT VARIABLES")
    print("=" * 70)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        print("Constant columns:")
        for column in constant_columns:
            print(f"  - {column}")
    else:
        print("No constant columns found.")


# ============================================================
# Unique values
# ============================================================

def print_unique_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("UNIQUE VALUE COUNTS")
    print("=" * 70)

    for column in df.columns:
        print(f"{column:10} : {df[column].nunique(dropna=False)}")


# ============================================================
# Treatment distribution
# ============================================================

def print_treatment_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TREATMENT DISTRIBUTION - trt")
    print("=" * 70)

    counts = df["trt"].value_counts(dropna=False).sort_index()
    percentages = (
        df["trt"]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        * 100
    )

    report = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    print(report.to_string())


# ============================================================
# Outcome distribution
# ============================================================

def print_outcome_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("OUTCOME DISTRIBUTION - label")
    print("=" * 70)

    counts = df["label"].value_counts(dropna=False).sort_index()
    percentages = (
        df["label"]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        * 100
    )

    report = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    print(report.to_string())


# ============================================================
# Numerical summary
# ============================================================

def print_numerical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NUMERICAL SUMMARY")
    print("=" * 70)

    print(df.describe().T.to_string())


# ============================================================
# Categorical / low-cardinality variables
# ============================================================

def print_categorical_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CATEGORICAL / LOW-CARDINALITY VARIABLES")
    print("=" * 70)

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 10:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).sort_index().to_string())


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("ACTG175 DATASET INSPECTION")
    print("=" * 70)

    df = load_dataset()

    print_dataset_overview(df)
    print_data_types(df)
    print_missing_values(df)
    print_duplicates(df)
    print_constant_columns(df)
    print_unique_values(df)
    print_treatment_distribution(df)
    print_outcome_distribution(df)
    print_numerical_summary(df)
    print_categorical_values(df)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()