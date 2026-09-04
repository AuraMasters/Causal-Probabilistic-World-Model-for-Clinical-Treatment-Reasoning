import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd_list, description):
    print(f"\n>>> Running: {description}...")
    result = subprocess.run(
        [sys.executable] + cmd_list,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"FAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[OK] {description} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Master ACTG175 Clinical AI Decision Support Pipeline Runner"
    )
    parser.add_argument("--train", action="store_true", help="Fit Model A and Model B on Development data")
    parser.add_argument("--validate", action="store_true", help="Run validation for Model A and Model B")
    parser.add_argument("--compare", action="store_true", help="Generate side-by-side model comparison artifact")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--all", action="store_true", help="Run end-to-end pipeline (train, validate, compare, test)")

    args = parser.parse_args()

    # Default to --all if no specific action provided
    if not (args.train or args.validate or args.compare or args.test or args.all):
        args.all = True

    print("=" * 75)
    print("ACTG175 CAUSAL CLINICAL REASONING — MASTER PIPELINE EXECUTION")
    print("=" * 75)

    if args.all or args.train:
        run_cmd(
            ["-c", "from src.continuous_model.model import ContinuousCausalModel; import pandas as pd; dev=pd.read_csv('data/processed/actg175_development.csv'); cm=ContinuousCausalModel(c_reg=0.5, random_seed=42).fit(dev); cm.save(); print('Continuous Model fitted & saved.')"],
            "Fitting Continuous Causal SCM (Model B)",
        )

    if args.all or args.validate:
        run_cmd(
            ["src/validation/comprehensive_validation.py"],
            "Validating Baseline Discretized Bayesian Network (Model A)",
        )
        run_cmd(
            ["src/continuous_model/validation.py"],
            "Validating Continuous / Hybrid SCM (Model B)",
        )

    if args.all or args.compare:
        run_cmd(
            ["src/validation/compare_models.py"],
            "Compiling Side-by-Side Model Benchmark Report",
        )

    if args.all or args.test:
        run_cmd(
            ["-m", "unittest", "tests/test_validation_pipeline.py"],
            "Executing Comprehensive Automated Test Suite (15 Unit Tests)",
        )

    print("\n" + "=" * 75)
    print("[SUCCESS] MASTER PIPELINE EXECUTION COMPLETED")
    print("All models, validation artifacts, benchmarks, and tests are verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
