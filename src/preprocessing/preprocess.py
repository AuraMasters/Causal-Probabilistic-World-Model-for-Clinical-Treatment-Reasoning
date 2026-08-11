from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "AIDS_ClinicalTrial_GroupStudy175.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEVELOPMENT_PATH = PROCESSED_DIR / "actg175_development.csv"
TEST_PATH = PROCESSED_DIR / "actg175_test.csv"
METADATA_PATH = PROCESSED_DIR / "preprocessing_metadata.json"


# ============================================================
# Approved variable groups
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

DECISION_VARIABLE = "trt"

OUTCOME_VARIABLE = "label"

SUPPORTING_VARIABLES = [
    "time",
]

EXCLUDED_VARIABLES = [
    "zprior",
    "str2",
    "treat",
    "cd420",
    "cd820",
    "offtrt",
]


# ============================================================
# Expected raw schema
# ============================================================

EXPECTED_COLUMNS = [
    "time",
    "trt",
    "age",
    "wtkg",
    "hemo",
    "homo",
    "drugs",
    "karnof",
    "oprior",
    "z30",
    "zprior",
    "preanti",
    "race",
    "gender",
    "str2",
    "strat",
    "symptom",
    "treat",
    "offtrt",
    "cd40",
    "cd420",
    "cd80",
    "cd820",
    "label",
]


# ============================================================
# Validation
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Validate the raw ACTG175 schema."""

    actual_columns = df.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset schema does not match the expected ACTG175 schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    if len(df) != 2139:
        raise ValueError(
            f"Expected 2139 rows, but found {len(df)}."
        )

    if df.isnull().sum().sum() != 0:
        raise ValueError(
            "Missing values detected in raw dataset."
        )

    if df.duplicated().sum() != 0:
        raise ValueError(
            "Duplicate rows detected in raw dataset."
        )


# ============================================================
# Create modeling dataframe
# ============================================================

def create_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the Version-1 modeling dataframe.

    Keeps:
        baseline variables
        treatment decision
        outcome

    Does not include:
        supporting time
        excluded/post-treatment variables
    """

    modeling_columns = (
        BASELINE_VARIABLES
        + [DECISION_VARIABLE]
        + [OUTCOME_VARIABLE]
    )

    return df[modeling_columns].copy()


# ============================================================
# Main preprocessing
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ACTG175 PHASE-1 PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    print(f"\nRaw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Validate raw dataset
    # --------------------------------------------------------

    validate_schema(df)

    print("Raw schema validation: PASSED")

    # --------------------------------------------------------
    # Create modeling dataset
    # --------------------------------------------------------

    modeling_df = create_modeling_dataframe(df)

    print(
        f"Modeling dataset shape: {modeling_df.shape}"
    )

    print("\nModeling variables:")

    for column in modeling_df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Split development/test
    # --------------------------------------------------------

    development_df, test_df = train_test_split(
        modeling_df,
        test_size=0.20,
        random_state=42,
        stratify=modeling_df[
            [DECISION_VARIABLE, OUTCOME_VARIABLE]
        ].astype(str).agg("_".join, axis=1),
    )

    print(
        f"\nDevelopment set: {development_df.shape}"
    )

    print(
        f"Test set:        {test_df.shape}"
    )

    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_df.to_csv(
        DEVELOPMENT_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "raw_dataset": str(
            RAW_PATH.relative_to(PROJECT_ROOT)
        ),
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
        "development_rows": int(len(development_df)),
        "test_rows": int(len(test_df)),
        "random_state": 42,
        "test_size": 0.20,
        "baseline_variables": BASELINE_VARIABLES,
        "decision_variable": DECISION_VARIABLE,
        "outcome_variable": OUTCOME_VARIABLE,
        "supporting_variables": SUPPORTING_VARIABLES,
        "excluded_variables": EXCLUDED_VARIABLES,
        "note": (
            "Continuous variables are retained in their original "
            "form at Phase 1. Discretization is performed separately "
            "in Phase 2 using development data only."
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nSaved:")
    print(f"  {DEVELOPMENT_PATH}")
    print(f"  {TEST_PATH}")
    print(f"  {METADATA_PATH}")

    print("\n" + "=" * 70)
    print("PHASE-1 PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()