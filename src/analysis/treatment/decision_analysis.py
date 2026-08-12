from pathlib import Path
import json
import pandas as pd


# ============================================================
# ACTG175 PHASE-19
# DECISION / UTILITY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INTERVENTION_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
    / "intervention"
    / "intervention_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
    / "decision"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SIMPLE TRANSPARENT UTILITY MODEL
# ============================================================

UTILITY_LABEL_0 = 1.0
UTILITY_LABEL_1 = 0.0


# ============================================================
# LOAD INTERVENTION RESULTS
# ============================================================

def load_results():

    if not INTERVENTION_RESULTS.exists():
        raise FileNotFoundError(
            "Phase-18 intervention results not found:\n"
            f"{INTERVENTION_RESULTS}\n\n"
            "Run Phase 18 first."
        )

    data = pd.read_csv(
        INTERVENTION_RESULTS
    )

    required_columns = {
        "treatment",
        "treatment_name",
        "p_label_0",
        "p_label_1",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + ", ".join(sorted(missing))
        )

    if len(data) != 4:
        raise ValueError(
            f"Expected 4 treatments, found {len(data)}."
        )

    return data


# ============================================================
# CALCULATE EXPECTED UTILITY
# ============================================================

def calculate_utility(data):

    data = data.copy()

    data["utility_label_0"] = (
        UTILITY_LABEL_0
    )

    data["utility_label_1"] = (
        UTILITY_LABEL_1
    )

    data["expected_utility"] = (
        data["p_label_0"]
        * UTILITY_LABEL_0
        +
        data["p_label_1"]
        * UTILITY_LABEL_1
    )

    # Higher utility is better.
    data["utility_rank"] = (
        data["expected_utility"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    best_index = (
        data["expected_utility"]
        .idxmax()
    )

    data["is_best_treatment"] = False

    data.loc[
        best_index,
        "is_best_treatment",
    ] = True

    return data


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(data):

    print()
    print("=" * 70)
    print("PHASE-19 DECISION / UTILITY ANALYSIS")
    print("=" * 70)

    print()
    print("Utility definition:")
    print(
        f"Label 0 utility = {UTILITY_LABEL_0:.1f}"
    )
    print(
        f"Label 1 utility = {UTILITY_LABEL_1:.1f}"
    )

    print()
    print(
        "Expected Utility = "
        "P(Label 0) × Utility(Label 0) + "
        "P(Label 1) × Utility(Label 1)"
    )

    print()
    print("=" * 70)
    print("TREATMENT UTILITIES")
    print("=" * 70)

    for _, row in data.iterrows():

        print()
        print(
            f"Treatment {int(row['treatment'])}"
        )

        print(
            f"  {row['treatment_name']}"
        )

        print(
            f"  P(Label 0): "
            f"{row['p_label_0']:.6f}"
        )

        print(
            f"  P(Label 1): "
            f"{row['p_label_1']:.6f}"
        )

        print(
            f"  Expected Utility: "
            f"{row['expected_utility']:.6f}"
        )

        print(
            f"  Utility Rank: "
            f"{int(row['utility_rank'])}"
        )

    best = data.loc[
        data["is_best_treatment"]
    ].iloc[0]

    print()
    print("=" * 70)
    print("BEST TREATMENT UNDER UTILITY MODEL")
    print("=" * 70)

    print()
    print(
        f"Treatment {int(best['treatment'])}"
    )

    print(
        best["treatment_name"]
    )

    print()
    print(
        f"Expected Utility: "
        f"{best['expected_utility']:.6f}"
    )

    print(
        f"P(Label 0): "
        f"{best['p_label_0']:.2%}"
    )

    print(
        f"P(Label 1): "
        f"{best['p_label_1']:.2%}"
    )

    print()
    print(
        "Decision rule:"
    )

    print(
        "Choose the treatment with the "
        "highest expected utility."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This utility function is a simple "
        "transparent model assumption."
    )

    print(
        "It is NOT a clinically validated "
        "utility scale."
    )

    print(
        "The result is therefore a "
        "model-based decision, not a "
        "medical prescription."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(data):

    results_path = (
        OUTPUT_DIR
        / "decision_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "decision_summary.json"
    )

    data.to_csv(
        results_path,
        index=False,
    )

    best = data.loc[
        data["is_best_treatment"]
    ].iloc[0]

    summary = {

        "phase": 19,

        "analysis": (
            "Decision and expected utility analysis"
        ),

        "utility_model": {

            "label_0_utility":
                UTILITY_LABEL_0,

            "label_1_utility":
                UTILITY_LABEL_1,

            "formula":
                "EU = P(label=0)*U0 + "
                "P(label=1)*U1",
        },

        "decision_rule": (
            "Select treatment with the "
            "highest expected utility."
        ),

        "best_treatment": {

            "trt":
                int(best["treatment"]),

            "name":
                best["treatment_name"],

            "expected_utility":
                float(best["expected_utility"]),

            "p_label_0":
                float(best["p_label_0"]),

            "p_label_1":
                float(best["p_label_1"]),
        },

        "clinical_warning": (
            "The utility values are simplified "
            "research assumptions and are not "
            "clinically validated."
        ),
    }

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

    print()
    print(
        "Saved decision results:"
    )

    print(
        results_path
    )

    print()
    print(
        "Saved decision summary:"
    )

    print(
        summary_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-19")
    print("DECISION / UTILITY ANALYSIS")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Phase 18 intervention results are "
        "used as input."
    )

    print(
        "The final 23-edge DAG is NOT modified."
    )

    print(
        "No new parameters are learned."
    )

    print()
    print(
        "Loading Phase-18 results..."
    )

    data = load_results()

    print(
        "Phase-18 results: READY"
    )

    print(
        f"Treatments found: {len(data)}"
    )

    print()
    print(
        "Calculating expected utilities..."
    )

    data = calculate_utility(
        data
    )

    print(
        "Utility calculation: PASSED"
    )

    display_results(
        data
    )

    save_results(
        data
    )

    print()
    print("=" * 70)
    print("PHASE-19 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()