from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actg175_development.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actg175_test.csv"
)


EXPECTED_COLUMNS = [
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


def inspect_dataset(name: str, path: Path) -> None:

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    print("\nSchema correct:", df.columns.tolist() == EXPECTED_COLUMNS)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print(
        "Missing values:",
        int(df.isnull().sum().sum())
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    print(
        "Duplicate rows:",
        int(df.duplicated().sum())
    )

    # --------------------------------------------------------
    # Treatment distribution
    # --------------------------------------------------------

    print("\nTreatment distribution:")
    treatment = (
        df["trt"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        pd.DataFrame(
            {
                "count": df["trt"].value_counts().sort_index(),
                "percentage": treatment.round(2),
            }
        )
    )

    # --------------------------------------------------------
    # Outcome distribution
    # --------------------------------------------------------

    print("\nOutcome distribution:")
    outcome = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        pd.DataFrame(
            {
                "count": df["label"].value_counts().sort_index(),
                "percentage": outcome.round(2),
            }
        )
    )

    # --------------------------------------------------------
    # Treatment × Outcome
    # --------------------------------------------------------

    print("\nTreatment × Outcome:")
    print(
        pd.crosstab(
            df["trt"],
            df["label"],
            margins=True,
        )
    )

    # --------------------------------------------------------
    # Treatment × Outcome percentages
    # --------------------------------------------------------

    print("\nTreatment × Outcome percentages:")
    print(
        (
            pd.crosstab(
                df["trt"],
                df["label"],
                normalize="index",
            )
            * 100
        ).round(2)
    )


def main() -> None:

    print("=" * 70)
    print("ACTG175 PROCESSED DATA VERIFICATION")
    print("=" * 70)

    inspect_dataset(
        "DEVELOPMENT DATASET",
        DEVELOPMENT_PATH,
    )

    inspect_dataset(
        "TEST DATASET",
        TEST_PATH,
    )

    print("\n" + "=" * 70)
    print("PROCESSED DATA VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()