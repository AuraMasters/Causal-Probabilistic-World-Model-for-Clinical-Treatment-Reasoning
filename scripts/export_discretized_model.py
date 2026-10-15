"""
Exports Model A (Discretized Bayesian Network) CPT and metadata
to frontend/src/lib/discretizedModelData.json for exact client-side inference.
"""
import json
import math
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.service import (
    get_discretized_model,
    get_metadata,
    intervention_probability,
)


def export():
    model = get_discretized_model()
    meta = get_metadata()

    cpt_path = PROJECT_ROOT / "results" / "final_model" / "cpts" / "label_cpt.csv"
    cpt_df = pd.read_csv(cpt_path)
    p1_df = cpt_df[cpt_df["label"] == 1]

    cpt_map = {}
    for _, row in p1_df.iterrows():
        key = f"{row['cd40']}_{row['trt']}_{row['z30']}"
        cpt_map[key] = round(float(row["probability"]), 6)

    ablation_no_cd40 = {}
    for trt in ["0", "1", "2", "3"]:
        for z in ["0", "1"]:
            _, p1 = intervention_probability(model, {"z30": z}, trt)
            ablation_no_cd40[f"{trt}_{z}"] = round(float(p1), 6)

    ablation_no_z30 = {}
    for trt in ["0", "1", "2", "3"]:
        for c in ["cd40_1", "cd40_2", "cd40_3"]:
            _, p1 = intervention_probability(model, {"cd40": c}, trt)
            ablation_no_z30[f"{c}_{trt}"] = round(float(p1), 6)

    clean_variables = {}
    for var, v_info in meta["variables"].items():
        clean_edges = []
        for e in v_info["edges"]:
            try:
                val = float(e)
                clean_edges.append(val if math.isfinite(val) else None)
            except (TypeError, ValueError):
                clean_edges.append(None)
        clean_variables[var] = {
            **v_info,
            "edges": clean_edges,
        }

    payload = {
        "model_name": "Model A: Baseline Discretized Bayesian Network",
        "representation": "sparse",
        "variables": clean_variables,
        "cpt_label_1": cpt_map,
        "ablation_no_cd40": ablation_no_cd40,
        "ablation_no_z30": ablation_no_z30,
    }

    out_path = PROJECT_ROOT / "frontend" / "src" / "lib" / "discretizedModelData.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Exported discretizedModelData.json successfully to {out_path} ({len(cpt_map)} CPT entries).")


if __name__ == "__main__":
    export()
