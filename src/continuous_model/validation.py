import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continuous_model.model import (
    ALL_FEATURES,
    CONTINUOUS_VARIABLES,
    TARGET,
    TREATMENT,
    TREATMENTS,
    ContinuousCausalModel,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = PROJECT_ROOT / "data" / "processed" / "actg175_development.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "actg175_test.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "validation" / "continuous"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 1000


def run_continuous_validation():
    print("=" * 75)
    print("CONTINUOUS / HYBRID CAUSAL MODEL (MODEL B) VALIDATION PIPELINE")
    print("=" * 75)

    print("\n1. Loading continuous un-binned datasets...")
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for col in CONTINUOUS_VARIABLES:
        dev_df[col] = dev_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)
    for col in ALL_FEATURES[len(CONTINUOUS_VARIABLES):]:
        dev_df[col] = dev_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    print(f"   Development: {len(dev_df)} rows | Test: {len(test_df)} rows")
    print(f"   Continuous Variables ({len(CONTINUOUS_VARIABLES)}): {', '.join(CONTINUOUS_VARIABLES)}")

    print("\n2. Fitting ContinuousCausalModel strictly on Development data...")
    model = ContinuousCausalModel(c_reg=0.5, random_seed=RANDOM_SEED)
    model.fit(dev_df)
    model.save()
    print("   Continuous StandardScaler and arm-specific response surfaces fitted.")

    print("\n3. Performing Development-Only Threshold Optimization...")
    # Compute development predictions under observed treatments
    X_dev = model.preprocessor.transform(dev_df[ALL_FEATURES])
    y_dev = dev_df[TARGET].astype(int).to_numpy()
    trt_dev = dev_df[TREATMENT].astype(int).to_numpy()

    dev_probs = np.zeros(len(dev_df))
    for i in range(len(dev_df)):
        t = trt_dev[i]
        dev_probs[i] = model.arm_models[t].predict_proba(X_dev[i : i + 1])[0, 1]

    dev_sweep = []
    best_j = -1.0
    best_tau = 0.50
    for tau in np.arange(0.05, 0.61, 0.01):
        tau = round(float(tau), 2)
        pred_dev = (dev_probs >= tau).astype(int)
        cm_d = confusion_matrix(y_dev, pred_dev, labels=[0, 1])
        tn_d, fp_d, fn_d, tp_d = cm_d.ravel()
        sens_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
        spec_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0.0
        prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
        f1_d = (2 * prec_d * sens_d) / (prec_d + sens_d) if (prec_d + sens_d) > 0 else 0.0
        acc_d = (tp_d + tn_d) / len(y_dev)
        j_d = sens_d + spec_d - 1.0

        item = {
            "threshold": tau,
            "sensitivity_recall": round(float(sens_d), 4),
            "specificity": round(float(spec_d), 4),
            "precision": round(float(prec_d), 4),
            "f1_score": round(float(f1_d), 4),
            "accuracy": round(float(acc_d), 4),
            "balanced_accuracy": round(float((sens_d + spec_d) / 2), 4),
            "youden_j": round(float(j_d), 4),
            "true_positives": int(tp_d),
            "false_positives": int(fp_d),
            "true_negatives": int(tn_d),
            "false_negatives": int(fn_d),
        }
        dev_sweep.append(item)
        if j_d > best_j:
            best_j = j_d
            best_tau = tau

    print(f"   Selected Development Optimal Decision Threshold: tau* = {best_tau:.2f} (Youden J = {best_j:.4f})")

    print("\n4. Running inference on 428 HELD-OUT test patients with exact continuous inputs...")
    X_test = model.preprocessor.transform(test_df[ALL_FEATURES])
    y_test = test_df[TARGET].astype(int).to_numpy()
    trt_test = test_df[TREATMENT].astype(int).to_numpy()

    test_probs = np.zeros(len(test_df))
    all_arm_test_probs = []

    for i in range(len(test_df)):
        t_obs = trt_test[i]
        p_obs = float(model.arm_models[t_obs].predict_proba(X_test[i : i + 1])[0, 1])
        test_probs[i] = p_obs

        arm_p = {}
        for t in [0, 1, 2, 3]:
            arm_p[t] = float(model.arm_models[t].predict_proba(X_test[i : i + 1])[0, 1])

        all_arm_test_probs.append({
            "observed_trt": t_obs,
            "actual_label": int(y_test[i]),
            "arm_probs": arm_p,
        })

    # Predictive metrics at default 0.50
    pred_05 = (test_probs >= 0.50).astype(int)
    cm_05 = confusion_matrix(y_test, pred_05, labels=[0, 1])
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()

    # Predictive metrics at calibrated best_tau
    pred_cal = (test_probs >= best_tau).astype(int)
    cm_cal = confusion_matrix(y_test, pred_cal, labels=[0, 1])
    tn_cal, fp_cal, fn_cal, tp_cal = cm_cal.ravel()

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in range(0, len(fpr), max(1, len(fpr)//35))]
    if roc_points[-1]["fpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_probs)
    pr_points = [{"recall": round(float(rec_curve[i]), 4), "precision": round(float(prec_curve[i]), 4)} for i in range(0, len(prec_curve), max(1, len(prec_curve)//35))]

    pred_metrics = {
        "accuracy": float(accuracy_score(y_test, pred_05)),
        "precision": float(precision_score(y_test, pred_05, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, pred_05, zero_division=0)),
        "specificity": float(tn_05 / (tn_05 + fp_05)),
        "f1_score": float(f1_score(y_test, pred_05, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "log_loss": float(log_loss(y_test, test_probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, test_probs)),
        "confusion_matrix": {
            "true_negatives": int(tn_05),
            "false_positives": int(fp_05),
            "false_negatives": int(fn_05),
            "true_positives": int(tp_05),
            "total": len(y_test),
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }

    test_sweep = []
    for tau in np.arange(0.05, 0.61, 0.01):
        tau = round(float(tau), 2)
        pred_t = (test_probs >= tau).astype(int)
        cm_t = confusion_matrix(y_test, pred_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        f1_t = (2 * prec_t * sens_t) / (prec_t + sens_t) if (prec_t + sens_t) > 0 else 0.0
        acc_t = (tp_t + tn_t) / len(y_test)
        j_t = sens_t + spec_t - 1.0

        test_sweep.append({
            "threshold": tau,
            "sensitivity_recall": round(float(sens_t), 4),
            "specificity": round(float(spec_t), 4),
            "precision": round(float(prec_t), 4),
            "f1_score": round(float(f1_t), 4),
            "accuracy": round(float(acc_t), 4),
            "balanced_accuracy": round(float((sens_t + spec_t) / 2), 4),
            "youden_j": round(float(j_t), 4),
            "true_positives": int(tp_t),
            "false_positives": int(fp_t),
            "true_negatives": int(tn_t),
            "false_negatives": int(fn_t),
        })

    threshold_analysis = {
        "investigation_summary": {
            "root_cause_of_low_sensitivity_at_0_5": "Preserving continuous measurements yields smoother posteriors; rigid 0.50 cutoff requires extreme risk; base rate is 24.3%.",
            "selection_criterion": "Maximum Youden's J statistic swept strictly on Development partition.",
            "optimal_threshold_tau": best_tau,
            "development_metrics_at_optimal": next(i for i in dev_sweep if i["threshold"] == best_tau),
            "test_default_threshold_0_50": next(i for i in test_sweep if i["threshold"] == 0.50),
            "test_calibrated_threshold": next(i for i in test_sweep if i["threshold"] == best_tau),
        },
        "development_threshold_sweep": dev_sweep,
        "test_threshold_sweep": test_sweep,
    }

    print("\n5. Computing bootstrap 95% confidence intervals (B=1,000 resamples)...")
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_test)
    boot_metrics = {k: [] for k in ["roc_auc", "pr_auc", "accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "brier_score", "log_loss"]}

    for _ in range(BOOTSTRAP_ROUNDS):
        idx_b = rng.choice(n, size=n, replace=True)
        yt_b = y_test[idx_b]
        yp_b = test_probs[idx_b]
        pred_b = (yp_b >= 0.5).astype(int)

        if len(np.unique(yt_b)) < 2:
            continue

        boot_metrics["roc_auc"].append(float(roc_auc_score(yt_b, yp_b)))
        boot_metrics["pr_auc"].append(float(average_precision_score(yt_b, yp_b)))
        boot_metrics["accuracy"].append(float(accuracy_score(yt_b, pred_b)))
        boot_metrics["precision"].append(float(precision_score(yt_b, pred_b, zero_division=0)))
        boot_metrics["recall_sensitivity"].append(float(recall_score(yt_b, pred_b, zero_division=0)))
        cm_b = confusion_matrix(yt_b, pred_b, labels=[0, 1])
        tn_b, fp_b, _, _ = cm_b.ravel()
        boot_metrics["specificity"].append(float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0)
        boot_metrics["f1_score"].append(float(f1_score(yt_b, pred_b, zero_division=0)))
        boot_metrics["brier_score"].append(float(brier_score_loss(yt_b, yp_b)))
        boot_metrics["log_loss"].append(float(log_loss(yt_b, yp_b, labels=[0, 1])))

    cis = {}
    for k, vals in boot_metrics.items():
        arr = np.asarray(vals, dtype=float)
        cis[k] = {
            "point_estimate": round(float(pred_metrics[k]), 4),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
            "std_error": round(float(np.std(arr)), 4),
        }

    print("\n6. Performing Calibration & Reliability Curve Analysis...")
    n_bins = 10
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        low = boundaries[i]
        hi = boundaries[i + 1]
        mask = (test_probs >= low) & (test_probs <= hi if i == n_bins - 1 else test_probs < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bins_data.append({"bin": i + 1, "lower": round(low, 2), "upper": round(hi, 2), "count": 0, "mean_predicted": None, "observed_rate": None, "absolute_gap": None})
            continue
        m_pred = float(np.mean(test_probs[mask]))
        obs_rate = float(np.mean(y_test[mask]))
        gap = abs(m_pred - obs_rate)
        ece += (cnt / n) * gap
        if gap > mce:
            mce = gap
        bins_data.append({"bin": i + 1, "lower": round(low, 2), "upper": round(hi, 2), "count": cnt, "mean_predicted": round(m_pred, 4), "observed_rate": round(obs_rate, 4), "absolute_gap": round(gap, 4)})

    eps = 1e-6
    clipped = np.clip(test_probs, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calib_lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    calib_lr.fit(logits, y_test)
    calib_intercept = round(float(calib_lr.intercept_[0]), 4)
    calib_slope = round(float(calib_lr.coef_[0][0]), 4)

    dist_0 = []
    dist_1 = []
    for i in range(n_bins):
        low = boundaries[i]
        hi = boundaries[i + 1]
        mask_0 = (y_test == 0) & ((test_probs >= low) & (test_probs <= hi if i == n_bins - 1 else test_probs < hi))
        mask_1 = (y_test == 1) & ((test_probs >= low) & (test_probs <= hi if i == n_bins - 1 else test_probs < hi))
        dist_0.append(int(np.sum(mask_0)))
        dist_1.append(int(np.sum(mask_1)))

    calib_analysis = {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier_score_loss(y_test, test_probs)), 4),
        "calibration_intercept": calib_intercept,
        "calibration_slope": calib_slope,
        "calibration_interpretation": f"ECE is {round(ece*100, 2)}%. Intercept = {calib_intercept} (ideal 0.0), Slope = {calib_slope} (ideal 1.0).",
        "bins": bins_data,
        "probability_distribution": {
            "bin_labels": [f"{round(boundaries[i], 1)}-{round(boundaries[i+1], 1)}" for i in range(n_bins)],
            "label_0_counts": dist_0,
            "label_1_counts": dist_1,
        },
    }

    print("\n7. Performing Decision Curve Analysis (DCA)...")
    dca_points = []
    pt_sweep = np.arange(0.05, 0.51, 0.025)
    for pt in pt_sweep:
        pt = round(float(pt), 4)
        wt = pt / (1.0 - pt)
        y_p = (test_probs >= pt).astype(int)
        tp_m = int(np.sum((y_test == 1) & (y_p == 1)))
        fp_m = int(np.sum((y_test == 0) & (y_p == 1)))
        nb_m = (tp_m / n) - (fp_m / n) * wt

        tp_a = int(np.sum(y_test == 1))
        fp_a = int(np.sum(y_test == 0))
        nb_a = (tp_a / n) - (fp_a / n) * wt

        avoided = (nb_m - nb_a) / wt * 100.0 if nb_a > 0 else max(0.0, nb_m / wt * 100.0)
        dca_points.append({
            "threshold_probability": pt,
            "net_benefit_model": round(float(nb_m), 4),
            "net_benefit_all": round(float(nb_a), 4),
            "net_benefit_none": 0.0,
            "interventions_avoided_per_100": round(float(avoided), 2),
        })

    superior_pts = [p["threshold_probability"] for p in dca_points if p["net_benefit_model"] > max(p["net_benefit_all"], 0.0)]
    sup_span = f"{round(min(superior_pts)*100, 1)}% to {round(max(superior_pts)*100, 1)}%" if superior_pts else "None"

    dca_analysis = {
        "event_prevalence": round(float(np.mean(y_test)), 4),
        "evaluation_cohort_size": n,
        "superior_threshold_range": sup_span,
        "interpretation": f"Continuous Model achieves positive Net Benefit superior to Treat-All/Treat-None from {sup_span}.",
        "dca_points": dca_points,
    }

    print("\n8. Evaluating Counterfactual Treatment Decisions...")
    better_c = 0
    same_c = 0
    worse_c = 0
    deltas = []
    patient_records = []

    for idx, item in enumerate(all_arm_test_probs):
        t_obs = item["observed_trt"]
        p_obs = item["arm_probs"][t_obs]
        t_rec = min(item["arm_probs"].keys(), key=lambda t: item["arm_probs"][t])
        p_rec = item["arm_probs"][t_rec]
        diff = p_obs - p_rec
        deltas.append(diff)

        if diff > 1e-6:
            better_c += 1
            status = "lower_risk"
        elif abs(diff) <= 1e-6:
            same_c += 1
            status = "same_risk"
        else:
            worse_c += 1
            status = "higher_risk"

        patient_records.append({
            "patient_idx": idx,
            "observed_treatment": t_obs,
            "observed_treatment_name": TREATMENTS[t_obs]["name"],
            "observed_risk": round(p_obs, 4),
            "recommended_treatment": t_rec,
            "recommended_treatment_name": TREATMENTS[t_rec]["name"],
            "recommended_risk": round(p_rec, 4),
            "risk_reduction": round(diff, 4),
            "status": status,
        })

    hist_c, bin_e = np.histogram(deltas, bins=[-0.01, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    cf_eval = {
        "test_patients": n,
        "better_count": better_c,
        "better_rate": round(better_c / n, 4),
        "same_count": same_c,
        "same_rate": round(same_c / n, 4),
        "worse_count": worse_c,
        "worse_rate": round(worse_c / n, 4),
        "mean_risk_reduction": round(float(np.mean(deltas)), 4),
        "median_risk_reduction": round(float(np.median(deltas)), 4),
        "max_risk_reduction": round(float(np.max(deltas)), 4),
        "risk_reduction_histogram": {
            "bins": [f"{round(bin_e[i]*100, 1)}%–{round(bin_e[i+1]*100, 1)}%" for i in range(len(hist_c))],
            "counts": [int(c) for c in hist_c],
        },
        "sample_patient_decisions": patient_records[:10],
    }

    print("\n9. Evaluating Clinical Subgroups on True Continuous Measurements...")
    subgroups_def = [
        ("Baseline CD4 < 200 (Severe)", test_df["cd40"] < 200),
        ("Baseline CD4 200–350 (Moderate)", (test_df["cd40"] >= 200) & (test_df["cd40"] <= 350)),
        ("Baseline CD4 > 350 (Mild/Preserved)", test_df["cd40"] > 350),
        ("Age < 35 Years", test_df["age"] < 35),
        ("Age 35–50 Years", (test_df["age"] >= 35) & (test_df["age"] <= 50)),
        ("Age > 50 Years", test_df["age"] > 50),
        ("Karnofsky < 90% (Reduced Function)", test_df["karnof"] < 90),
        ("Karnofsky ≥ 90% (Normal Function)", test_df["karnof"] >= 90),
        ("Asymptomatic at Baseline", test_df["symptom"] == 0),
        ("Symptomatic at Baseline", test_df["symptom"] == 1),
        ("ART Naive (No Prior ZDV)", test_df["z30"] == 0),
        ("ART Experienced (Prior ZDV)", test_df["z30"] == 1),
        ("Male Cohort", test_df["gender"] == 1),
        ("Female Cohort", test_df["gender"] == 0),
    ]

    subgroup_results = []
    for s_name, mask in subgroups_def:
        st = y_test[mask.to_numpy()]
        sp = test_probs[mask.to_numpy()]
        sz = len(st)
        if sz == 0:
            continue
        auc = round(float(roc_auc_score(st, sp)), 4) if len(np.unique(st)) >= 2 else None
        prauc = round(float(average_precision_score(st, sp)), 4) if len(np.unique(st)) >= 2 else None
        spred = (sp >= 0.5).astype(int)
        acc = round(float(accuracy_score(st, spred)), 4)
        prec = round(float(precision_score(st, spred, zero_division=0)), 4)
        rec = round(float(recall_score(st, spred, zero_division=0)), 4)
        f1_val = round(float(f1_score(st, spred, zero_division=0)), 4)
        brier_val = round(float(brier_score_loss(st, sp)), 4)
        cm_s = confusion_matrix(st, spred, labels=[0, 1])
        tn_s, fp_s, _, _ = cm_s.ravel()
        spec_val = round(float(tn_s / (tn_s + fp_s)), 4) if (tn_s + fp_s) > 0 else 0.0

        subgroup_results.append({
            "subgroup_name": s_name,
            "sample_size": sz,
            "positive_events": int(np.sum(st)),
            "event_rate": round(float(np.mean(st)), 4),
            "roc_auc": auc,
            "pr_auc": prauc,
            "accuracy": acc,
            "precision": prec,
            "sensitivity": rec,
            "specificity": spec_val,
            "f1_score": f1_val,
            "brier_score": brier_val,
            "is_reliable": bool(sz >= 30),
            "reliability_note": "Adequate sample size (N ≥ 30)" if sz >= 30 else "Small sample size (N < 30) — interpret with caution",
        })

    payload = {
        "model_architecture": "Continuous / Hybrid SCM (G-Computation)",
        "information_preservation": "Full continuous numerical measurements preserved without binning",
        "dataset": "ACTG175 Clinical Trial",
        "development_rows": len(dev_df),
        "test_rows": len(test_df),
        "continuous_variables": CONTINUOUS_VARIABLES,
        "categorical_variables": ALL_FEATURES[len(CONTINUOUS_VARIABLES):],
        "random_seed": RANDOM_SEED,
        "predictive_metrics": pred_metrics,
        "confidence_intervals_95": cis,
        "calibration": calib_analysis,
        "decision_curve_analysis": dca_analysis,
        "threshold_analysis": threshold_analysis,
        "counterfactual_treatment_evaluation": cf_eval,
        "subgroup_analysis": subgroup_results,
        "methodology_and_limitations": {
            "continuous_modeling": "Continuous biomarkers are standardized using development-only parameters and evaluated through arm-specific continuous response surfaces.",
            "data_partitioning": "Strict 80/20 partition (1,711 dev vs 428 test). Preprocessing transforms fitted exclusively on development rows.",
            "causal_assumptions": "Assumes trial conditional exchangeability, positivity, and consistency within the ACTG175 trial.",
            "clinical_disclaimer": "Research and educational decision-support prototype. Not approved for autonomous diagnostic or treatment decisions.",
        },
    }
    def to_serializable(val):
        if isinstance(val, dict):
            return {str(k): to_serializable(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [to_serializable(x) for x in val]
        elif isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating, float)):
            return float(val) if math.isfinite(float(val)) else None
        elif isinstance(val, (np.bool_, bool)):
            return bool(val)
        elif isinstance(val, np.ndarray):
            return to_serializable(val.tolist())
        return val

    out_file = OUTPUT_DIR / "continuous_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=4)

    print(f"\n[OK] Model B validation results saved to:\n   {out_file}")
    print("\nSummary of Continuous Model (Model B) Performance:")
    print(f"   ROC-AUC:  {pred_metrics['roc_auc']:.4f} (95% CI: {cis['roc_auc']['ci_lower']:.4f} - {cis['roc_auc']['ci_upper']:.4f})")
    print(f"   PR-AUC:   {pred_metrics['pr_auc']:.4f} (95% CI: {cis['pr_auc']['ci_lower']:.4f} - {cis['pr_auc']['ci_upper']:.4f})")
    print(f"   Accuracy: {pred_metrics['accuracy']*100:.2f}% (95% CI: {cis['accuracy']['ci_lower']*100:.2f}% - {cis['accuracy']['ci_upper']*100:.2f}%)")
    print(f"   Brier:    {pred_metrics['brier_score']:.4f} (95% CI: {cis['brier_score']['ci_lower']:.4f} - {cis['brier_score']['ci_upper']:.4f})")
    print(f"   ECE:      {calib_analysis['ece']:.4f}")
    print(f"   Optimal Threshold: tau* = {best_tau:.2f} (Sensitivity: {threshold_analysis['investigation_summary']['test_calibrated_threshold']['sensitivity_recall']*100:.1f}%, Specificity: {threshold_analysis['investigation_summary']['test_calibrated_threshold']['specificity']*100:.1f}%)")
    print(f"   Counterfactual Advantage Rate: {cf_eval['better_rate']*100:.1f}% ({cf_eval['better_count']}/{cf_eval['test_patients']})")
    print("=" * 75)


if __name__ == "__main__":
    run_continuous_validation()
