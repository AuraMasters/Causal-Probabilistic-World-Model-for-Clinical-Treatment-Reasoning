import pandas as pd
import numpy as np

def audit():
    df = pd.read_csv("data/raw/AIDS_ClinicalTrial_GroupStudy175.csv")
    print("=== ACTG 175 CLINICAL OUTCOME & CENSORING AUDIT ===")
    print("Total rows:", len(df))
    print("Overall event count (label=1):", int(df["label"].sum()), f"({df['label'].mean():.2%})")
    print("Overall censored count (label=0):", int((df["label"] == 0).sum()), f"({(df['label'] == 0).mean():.2%})")
    print()
    
    cens = df[df["label"] == 0]
    ev = df[df["label"] == 1]
    
    print("Event times (days):")
    print("  Min:", ev["time"].min(), "Median:", ev["time"].median(), "Mean:", round(ev["time"].mean(), 1), "Max:", ev["time"].max())
    print("Censoring times (days):")
    print("  Min:", cens["time"].min(), "Median:", cens["time"].median(), "Mean:", round(cens["time"].mean(), 1), "Max:", cens["time"].max())
    print()
    
    c_under_1yr = (cens["time"] < 365).sum()
    c_under_2yr = (cens["time"] < 730).sum()
    c_over_2yr = (cens["time"] >= 730).sum()
    print(f"Censored < 1 year (365 d): {c_under_1yr} ({c_under_1yr/len(cens):.1%})")
    print(f"Censored < 2 years (730 d): {c_under_2yr} ({c_under_2yr/len(cens):.1%})")
    print(f"Censored >= 2 years (730 d): {c_over_2yr} ({c_over_2yr/len(cens):.1%})")
    print()
    
    print("Breakdown by Treatment Arm:")
    for arm in [0, 1, 2, 3]:
        sub = df[df["trt"] == arm]
        sub_ev = sub[sub["label"] == 1]
        sub_cens = sub[sub["label"] == 0]
        print(f"Arm {arm}: N={len(sub)}")
        print(f"  Events: {len(sub_ev)} ({len(sub_ev)/len(sub):.1%}), median time: {sub_ev['time'].median():.0f} d")
        print(f"  Censored: {len(sub_cens)} ({len(sub_cens)/len(sub):.1%}), median time: {sub_cens['time'].median():.0f} d")
        print(f"  Early Censored (<730 d): {(sub_cens['time'] < 730).sum()} ({(sub_cens['time'] < 730).mean():.1%})")

if __name__ == "__main__":
    audit()
