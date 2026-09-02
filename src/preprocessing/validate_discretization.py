from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES = {
    "sparse": PROCESSED_DIR / "sparse",
    "rich": PROCESSED_DIR / "rich",
}

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

DISCRETIZED_VARIABLES = [
    "age",
    "wtkg",
    "karnof",
    "preanti",
    "cd40",
    "cd80",
]


def validate_dataset(name: str, path: Path) -> None:

    print("\n" + "=" * 70)
    print(f"{name.upper()} - {path.name}")
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    schema_ok = (
        df.columns.tolist() == EXPECTED_COLUMNS
    )

    print(f"Schema correct: {schema_ok}")

    if not schema_ok:
        raise ValueError(
            f"Incorrect schema in {path}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = int(
        df.isnull().sum().sum()
    )

    print(f"Missing values: {missing}")

    if missing != 0:
        raise ValueError(
            f"Missing values found in {path}"
        )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    collapsed_rows = int(
    df.duplicated().sum()
    )

    collapsed_percentage = (
    collapsed_rows / len(df) * 100
    )

    print(
        f"Collapsed discretized rows: "
        f"{collapsed_rows} "
        f"({collapsed_percentage:.2f}%)"
    )

    # --------------------------------------------------------
    # Variable states
    # --------------------------------------------------------

    print("\nState counts:")

    for column in DISCRETIZED_VARIABLES:

        counts = (
            df[column]
            .value_counts(dropna=False)
            .sort_index()
        )

        print(f"\n{column}:")
        print(counts.to_string())

        if df[column].nunique() < 2:
            print(
                f"WARNING: {column} has fewer than 2 states."
            )

    # --------------------------------------------------------
    # Treatment
    # --------------------------------------------------------

    print("\nTreatment states:")
    print(
        df["trt"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    print("\nOutcome states:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():

    print("=" * 70)
    print("ACTG175 DISCRETIZATION VALIDATION")
    print("=" * 70)

    for name, directory in CANDIDATES.items():

        development_path = (
            directory / "development.csv"
        )

        test_path = (
            directory / "test.csv"
        )

        if not development_path.exists():
            raise FileNotFoundError(
                development_path
            )

        if not test_path.exists():
            raise FileNotFoundError(
                test_path
            )

        validate_dataset(
            f"{name} development",
            development_path,
        )

        validate_dataset(
            f"{name} test",
            test_path,
        )

    print("\n" + "=" * 70)
    print("DISCRETIZATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()