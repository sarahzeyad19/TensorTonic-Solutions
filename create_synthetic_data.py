#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SYNTHETIC DATA GENERATOR FOR TESTING
================================================================================
Creates synthetic RUT and SCB data matching the structure of:
  Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx

Use this to test model_diagnostics_v6_wang_benchmark.py without real data.

Output: synthetic_KEEP_REPLICATES.xlsx (in current directory)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(42)

# ===================== CONFIG ======================
N_MIXES_RUT = 1640  # Unique mixes
REPS_PER_MIX_RUT = 2.8  # Average replicates
N_ROWS_RUT = int(N_MIXES_RUT * REPS_PER_MIX_RUT)  # ~4,600 rows

N_MIXES_SCB = 1271  # Unique mixes
REPS_PER_MIX_SCB = 2.4  # Average replicates
N_ROWS_SCB = int(N_MIXES_SCB * REPS_PER_MIX_SCB)  # ~3,050 rows

DESIGN_LEVELS = ["1", "1F", "2", "2F", "A", "Low ADT"]
MIX_TYPES = ["Wearing Course", "Binder Course", "Base Course", "Incidental"]

# ===================== HELPER FUNCTIONS ======================
def create_synthetic_features(n_rows):
    """Generate realistic feature distributions."""
    X = pd.DataFrame(index=range(n_rows))

    # Volumetric properties
    X["Va"] = norm.rvs(3.5, 1.0, n_rows).clip(0.5, 7.0)  # Void content
    X["VMA"] = norm.rvs(16.0, 2.0, n_rows).clip(10, 25)  # Void in mineral aggregate
    X["VFA"] = norm.rvs(70.0, 8.0, n_rows).clip(50, 90)  # Void filled with asphalt
    X["Pbe"] = X["VMA"] * X["VFA"] / 100  # Effective binder content
    X["Pba"] = norm.rvs(0.8, 0.2, n_rows).clip(0.2, 1.5)
    X["Gse"] = norm.rvs(2.68, 0.05, n_rows).clip(2.6, 2.8)  # Effective gravity

    # Aggregate properties
    X["Gsb"] = norm.rvs(2.50, 0.08, n_rows).clip(2.30, 2.70)  # Bulk gravity
    X["FAA"] = norm.rvs(45, 15, n_rows).clip(20, 75)  # Fine aggregate angularity
    X["CAA"] = norm.rvs(65, 12, n_rows).clip(40, 90)  # Coarse aggregate angularity
    X["SandEq"] = norm.rvs(65, 10, n_rows).clip(40, 85)
    X["FlatElong"] = norm.rvs(8, 4, n_rows).clip(1, 20)
    X["Absorption"] = norm.rvs(1.2, 0.5, n_rows).clip(0.2, 3.0)

    # Binder properties
    X["AC"] = norm.rvs(5.5, 0.6, n_rows).clip(3.5, 8.0)  # Asphalt content
    X["AC_from_RAP"] = X["AC"] * norm.rvs(0.3, 0.2, n_rows).clip(0, 1.0)
    X["Polymer"] = np.random.binomial(1, 0.3, n_rows)  # 30% polymer modified

    # Gradation (passing percentages)
    for sieve, pct in [("Pass_No_4", 90), ("Pass_No_8", 75), ("Pass_No_16", 60),
                        ("Pass_No_30", 40), ("Pass_No_50", 20), ("Pass_No_100", 8),
                        ("Pass_No_200", 3)]:
        X[sieve] = norm.rvs(pct, pct*0.15, n_rows).clip(0, 100)

    # Production context
    X["ADT"] = norm.rvs(8000, 6000, n_rows).clip(100, 30000)
    X["Production_Rate"] = norm.rvs(200, 80, n_rows).clip(50, 500)
    X["Mix_Temperature"] = norm.rvs(160, 15, n_rows).clip(130, 190)
    X["NMAS_mm"] = np.random.choice([9.5, 12.5, 19.0], n_rows)

    # PG grade (mostly filled/parsed)
    parsed_rate = 0.2
    X["PG_High"] = np.where(
        np.random.random(n_rows) < parsed_rate,
        np.random.choice([58, 64, 67, 70, 76, 82], n_rows),  # Parsed
        np.random.choice([64, 67, 76, 82], n_rows)  # Filled via rules
    )
    X["PG_Low"] = -22.0 + norm.rvs(0, 5, n_rows).clip(-15, -30)
    X["PG_span"] = X["PG_High"] - X["PG_Low"]
    X["PG_parsed"] = (np.random.random(n_rows) < parsed_rate).astype(float)

    # Engineered features
    X["RAP_Binder_Ratio"] = X["AC_from_RAP"] / X["AC"].replace(0, 5.0)
    X["Fine_Fraction"] = X["Pass_No_8"] - X["Pass_No_200"]
    X["Intermediate_Frac"] = X["Pass_No_4"] - X["Pass_No_8"]
    X["VMA_Filled_Index"] = X["VMA"] * X["VFA"] / 100
    sa = np.mean([0.41, 0.82, 1.64, 2.87, 6.14, 12.29, 32.77])
    X["AFT_micron"] = (X["Pbe"] / 100) / (sa * X["Gsb"].replace(0, 2.5)) * 1000

    # PG interactions
    X["PG_x_Va"] = X["PG_High"] * X["Va"]
    X["PG_x_ADT"] = X["PG_High"] * np.log1p(X["ADT"])
    X["PG_x_RBR"] = X["PG_High"] * X["RAP_Binder_Ratio"]
    X["PG_x_AFT"] = X["PG_High"] * X["AFT_micron"]

    # Categorical encoded
    X["Design_Level"] = np.random.choice(DESIGN_LEVELS, n_rows)
    X["Mix_Type"] = np.random.choice(MIX_TYPES, n_rows)

    # Handle any NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    return X

def create_group_ids(n_rows, n_unique_groups, avg_reps):
    """Create group assignments mimicking replicate structure."""
    groups = []
    remaining = n_rows
    for group_id in range(n_unique_groups):
        # Poisson-ish distribution of replicates
        n_reps = np.random.poisson(avg_reps)
        n_reps = max(1, min(n_reps, remaining))  # 1 to remaining
        groups.extend([f"Group_{group_id:06d}"] * n_reps)
        remaining -= n_reps
        if remaining == 0: break

    # Pad if needed
    while len(groups) < n_rows:
        groups.append(f"Group_{n_unique_groups:06d}")

    return np.array(groups[:n_rows])

def create_target(X, target_name, base_mean, base_std):
    """Generate correlated target variable from features."""
    # Synthetic model: target = f(features)
    if "RUT" in target_name:
        # RUT in mm: primarily affected by Va, PG_High, ADT, binder properties
        target = (
            4.0
            - 0.4 * (X["Va"] - 3.5)  # Lower Va -> more rutting
            - 0.2 * (X["PG_High"] - 67) * 0.05  # Higher PG -> less rutting
            + 0.3 * np.log1p(X["ADT"]) / 10  # Higher traffic -> more rutting
            + 0.15 * (X["AC"] - 5.5)  # Higher AC -> more rutting
            + norm.rvs(0, 0.5, len(X))  # Noise
        )
        target = target.clip(0.5, 10.0)  # Apply range limits
    else:  # SCB
        # SCB Jc in MPa√m: affected by VMA, gradation, binder
        target = (
            0.75
            + 0.05 * (X["VMA"] - 16)  # Higher VMA -> higher Jc
            + 0.02 * (X["Pass_No_200"] - 3)  # Fine content
            + 0.03 * (X["PG_High"] - 67) * 0.01  # PG effect
            - 0.02 * np.log1p(X["ADT"]) / 10  # Traffic reduces cracking resistance
            + norm.rvs(0, 0.1, len(X))
        )
        target = target.clip(0.3, 1.25)

    return target

# ===================== MAIN ======================
def create_workbook():
    print("Generating synthetic RUT data...")
    X_rut = create_synthetic_features(N_ROWS_RUT)
    grp_rut = create_group_ids(N_ROWS_RUT, N_MIXES_RUT, REPS_PER_MIX_RUT)
    y_rut = create_target(X_rut, "RUT", 4.0, 1.5)

    df_rut = X_rut.copy()
    df_rut["Exact_JMF_Group"] = grp_rut
    df_rut["LWT_Design_Result"] = y_rut
    df_rut["Custom_Name"] = [f"Mix_{i:05d}" for i in range(len(df_rut))]

    print(f"  {len(df_rut)} rows, {len(np.unique(grp_rut))} unique mixes, "
          f"target={y_rut.mean():.2f} ± {y_rut.std():.2f}")

    print("Generating synthetic SCB data...")
    X_scb = create_synthetic_features(N_ROWS_SCB)
    grp_scb = create_group_ids(N_ROWS_SCB, N_MIXES_SCB, REPS_PER_MIX_SCB)
    y_scb = create_target(X_scb, "SCB", 0.75, 0.25)

    df_scb = X_scb.copy()
    df_scb["Exact_JMF_Group"] = grp_scb
    df_scb["SCB_Result"] = y_scb
    df_scb["Custom_Name"] = [f"Mix_{i:05d}" for i in range(len(df_scb))]

    print(f"  {len(df_scb)} rows, {len(np.unique(grp_scb))} unique mixes, "
          f"target={y_scb.mean():.3f} ± {y_scb.std():.3f}")

    # Add design features for encoding
    for col in ["Design_Submission__Percent_Voids", "Design_Submission__VMA",
                "Design_Submission__VFA", "Design_Submission__Pbe",
                "Design_Submission__Pba", "Design_Submission__Gse",
                "Design_Submission__Percent_AC", "Design_Submission__Dust_Pbeff",
                "Design_Submission__Percent_Gmm_Ni", "Design_Submission__Percent_Gmm_Nm"]:
        col_short = col.split("__")[1]
        for df in [df_rut, df_scb]:
            if col_short in df.columns:
                df[col] = df[col_short]

    for col in ["Combined_Aggregate_Absorption", "Combined_Aggregate_FAA",
                "Combined_Aggregate_CAA", "Combined_Aggregate_Sand_Equivalent",
                "Combined_Aggregate_Flat_Elongated", "Combined_Aggregate_Bulk_Gravity"]:
        col_short = col.split("_")[-1]
        short_map = {"Absorption": "Absorption", "FAA": "FAA", "CAA": "CAA",
                     "Equivalent": "SandEq", "Elongated": "FlatElong",
                     "Gravity": "Gsb"}
        real_col = short_map.get(col_short, col_short)
        for df in [df_rut, df_scb]:
            if real_col in df.columns:
                df[col] = df[real_col]

    # Write to Excel
    output_file = "synthetic_KEEP_REPLICATES.xlsx"
    print(f"\nWriting to {output_file}...")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_rut.to_excel(writer, sheet_name="Rutting_Filtered", index=False)
        df_scb.to_excel(writer, sheet_name="SCB_Filtered", index=False)

    print(f"✓ Created {output_file}")
    print(f"  RUT sheet: {len(df_rut)} rows (target: {y_rut.mean():.2f} ± {y_rut.std():.2f})")
    print(f"  SCB sheet: {len(df_scb)} rows (target: {y_scb.mean():.3f} ± {y_scb.std():.3f})")
    print(f"\nTo test: python model_diagnostics_v6_wang_benchmark.py")

if __name__=="__main__":
    create_workbook()
