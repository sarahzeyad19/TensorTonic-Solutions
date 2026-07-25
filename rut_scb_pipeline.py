# -*- coding: utf-8 -*-
"""
Rutting (LWT) and SCB prediction  -  strong, leakage-safe pipeline
Approach:
    * build engineering feature set from the JMF design submission
    * remove duplicate EXACT mixes  ->  one row per Exact_JMF_Group
    * STRATIFIED, GROUP-aware splitting (no random split):
         - StratifiedGroupKFold holdout  (by Exact_JMF_Group, stratified on target bins)
         - StratifiedGroupKFold cross-validation
    * blended model  0.40 ExtraTrees + 0.35 LightGBM + 0.25 CatBoost
    * log target for rutting
Run directly in Spyder -> prints metrics and pops up all plots.

Requirements:  pip install pandas numpy scikit-learn lightgbm catboost matplotlib openpyxl

Reference results on the provided workbook (leakage-safe, grouped by exact mix):
    RUTTING : 1722 exact-mix groups | CV R2 0.69 +/- 0.05 | locked-test R2 0.77
    SCB     : 1316 exact-mix groups | CV R2 0.70 +/- 0.06 | locked-test R2 0.74
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap

# ------------------------------------------------------------------ CONFIG
# >>> EDIT THIS to the path of the workbook on your machine <<<
FILE = "Book1._rutting__scb_design_validation_xlsx.xlsx"
N_SPLITS   = 5
N_STRATA   = 5          # target quantile bins for stratification
RANDOM     = 42
OUTDIR     = "."        # where PNGs are also saved
HEADLESS   = False      # Spyder: keep False so plots pop up on screen

# ------------------------------------------------------------------ helpers
NMAS_MAP = {"1/2 in.": 12.5, "3/4 in.": 19.0, "1 in.": 25.0, "3/8 in.": 9.5}
# surface-area factors (m2/kg) per % passing, Superpave FA method
SA_FACTOR = {"Pass_No_4":0.41,"Pass_No_8":0.82,"Pass_No_16":1.64,"Pass_No_30":2.87,
             "Pass_No_50":6.14,"Pass_No_100":12.29,"Pass_No_200":32.77}

def num(s): return pd.to_numeric(s, errors="coerce")

def pick(df, *cands):
    for c in cands:
        if c in df.columns: return num(df[c])
    return pd.Series(np.nan, index=df.index)

def build_features(df):
    """canonical raw features + engineering features from a design sheet"""
    P = "Design_Submission__"; A = "Combined_Aggregate_"
    X = pd.DataFrame(index=df.index)
    # ---- volumetrics
    X["Va"]        = pick(df, P+"Percent_Voids")
    X["VMA"]       = pick(df, P+"VMA")
    X["VFA"]       = pick(df, P+"VFA")
    X["Pbe"]       = pick(df, P+"Pbe")
    X["Pba"]       = pick(df, P+"Pba")
    X["Gse"]       = pick(df, P+"Gse")
    X["Gmm"]       = pick(df, P+"Gmm")
    X["AC"]        = pick(df, P+"Percent_AC")
    X["Dust_Pbe"]  = pick(df, P+"Dust_Pbeff")
    X["Gmm_Ni"]    = pick(df, P+"Percent_Gmm_Ni")
    X["Gmm_Nm"]    = pick(df, P+"Percent_Gmm_Nm")
    # ---- aggregate
    X["Absorption"]= pick(df, A+"Absorption")
    X["FAA"]       = pick(df, A+"FAA")
    X["CAA"]       = pick(df, A+"CAA")
    X["SandEq"]    = pick(df, A+"Sand_Equivalent")
    X["FlatElong"] = pick(df, A+"Flat_Elongated")
    X["Gsb"]       = pick(df, A+"Bulk_Gravity")
    # ---- gradation
    sieves = ["Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4","Pass_No_8",
              "Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]
    for s in sieves: X[s] = pick(df, P+s)
    # ---- traffic / design
    X["ADT"]       = pick(df, "ADT")
    X["MixTemp"]   = pick(df, "Mix_Temperature")
    X["AC_from_RAP"]= pick(df, "Total_AC_From_RAP")
    X["NMAS_mm"]   = df["Nominal_Aggregate_Size"].map(NMAS_MAP)
    # categoricals -> ordinal codes
    for c in ["Design_Level","Mix_Type"]:
        X[c+"_code"] = df[c].astype("category").cat.codes

    # ---------------- ENGINEERING FEATURES ----------------
    X["RAP_Binder_Ratio"]   = X["AC_from_RAP"] / X["AC"].replace(0,np.nan)          # RBR proxy
    X["VirginBinder"]       = (X["AC"] - X["AC_from_RAP"]).clip(lower=0)
    X["Fines_to_Pbe"]       = X["Pass_No_200"] / X["Pbe"].replace(0,np.nan)
    X["Coarse_Fraction"]    = 100 - X["Pass_No_4"]
    X["Intermediate_Frac"]  = X["Pass_No_4"] - X["Pass_No_8"]
    X["Fine_Fraction"]      = X["Pass_No_8"] - X["Pass_No_200"]
    X["VMA_Filled_Index"]   = X["VMA"] * X["VFA"] / 100.0
    X["AbsorptionPenalty"]  = X["Absorption"] * X["AC"]
    X["PostDensification"]  = X["Gmm_Nm"] - 96.0                                     # %Gmm@Nmax margin
    X["InitialDensity"]     = X["Gmm_Ni"]
    # surface area (m2/kg) and asphalt film thickness proxy (micron)
    SA = sum(X[k]*v for k,v in SA_FACTOR.items()) / 100.0 + 0.41
    X["SurfaceArea"]        = SA
    X["AFT_micron"]         = (X["Pbe"]/100.0) / (SA * (X["Gsb"]).replace(0,np.nan)) * 1000.0
    X["CrackingExposure"]   = X["Va"] * SA / X["Pbe"].replace(0,np.nan)
    X["PG_proxy_x_Va"]      = X["MixTemp"] * X["Va"]      # temperature-void interaction
    return X

def make_dataset(sheet, target_col, ylo, yhi):
    df = pd.read_excel(FILE, sheet_name=sheet)
    y  = num(df[target_col])
    # EXACT-mix identity = mix identity + full rounded composition.
    #   identity columns stop different mixes collapsing together;
    #   composition columns merge true replicate specimens of the same JMF.
    ident = [c for c in ["Mix_ID","Design_Level"] if c in df.columns]
    allsieves = ["Pass_1_1_2in","Pass_1in","Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4",
                 "Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]
    comp = ["Design_Submission__Percent_AC","Design_Submission__Percent_Voids",
            "Design_Submission__VMA","Design_Submission__VFA","Design_Submission__Pbe",
            "Design_Submission__Gse"] + ["Design_Submission__"+s for s in allsieves] + \
           ["Nominal_Aggregate_Size","Mix_Type","Total_AC_From_RAP"]
    keycols = [c for c in ident + comp if c in df.columns]
    keysub = df[keycols].copy()
    for c in keycols:
        if keysub[c].dtype.kind in "fc":
            keysub[c] = num(keysub[c]).round(2)
    exact = keysub.apply(lambda col: col.map(str)).agg("|".join, axis=1)
    X = build_features(df)
    X["__y__"] = y
    X["__grp__"] = exact.values
    # drop missing target + winsorize to physically valid range (removes extraction errors)
    X = X[np.isfinite(X["__y__"]) & (X["__y__"] >= ylo) & (X["__y__"] <= yhi)]
    # ---- remove duplicate exact mixes : one row per Exact_JMF_Group (mean) ----
    agg = X.groupby("__grp__").mean(numeric_only=True)
    agg["__rep__"] = X.groupby("__grp__").size().values
    agg = agg.reset_index()
    y   = agg.pop("__y__")
    grp = agg.pop("__grp__")
    rep = agg.pop("__rep__")
    return agg, y.values, grp.values, rep.values

# ------------------------------------------------------------------ models
def make_models():
    et  = ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2,
                              random_state=RANDOM, n_jobs=-1)
    gbm = lgb.LGBMRegressor(n_estimators=900, learning_rate=0.02, num_leaves=31,
                            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                            random_state=RANDOM, n_jobs=-1, verbose=-1)
    cat = CatBoostRegressor(iterations=900, learning_rate=0.03, depth=6,
                            l2_leaf_reg=3.0, random_state=RANDOM, verbose=0)
    return {"ExtraTrees": et, "LightGBM": gbm, "CatBoost": cat}
WEIGHTS = {"ExtraTrees":0.40, "LightGBM":0.35, "CatBoost":0.25}

def blend(preds):  # preds: dict name->array
    return sum(WEIGHTS[k]*preds[k] for k in WEIGHTS)

def metrics(y, p):
    return dict(R2=r2_score(y,p),
                RMSE=np.sqrt(mean_squared_error(y,p)),
                MAE=mean_absolute_error(y,p))

# ------------------------------------------------------------------ core run
def run_target(name, sheet, target_col, log_target, ylo, yhi):
    print("\n" + "="*66 + f"\n  {name}\n" + "="*66)
    X, y, grp, rep = make_dataset(sheet, target_col, ylo, yhi)
    feat_names = list(X.columns)
    print(f"  exact-mix groups (rows after dedup): {len(X)}   "
          f"(avg replicates merged: {rep.mean():.2f})")
    print(f"  target  mean={y.mean():.3f}  std={y.std():.3f}  "
          f"range=[{y.min():.2f},{y.max():.2f}]")

    ytr_t = np.log1p(y) if log_target else y
    # stratify on target quantile bins
    strata = pd.qcut(y, N_STRATA, labels=False, duplicates="drop")

    # ---- STRATIFIED GROUP holdout : 1 fold (~1/N_SPLITS) as locked test ----
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM)
    tr_idx, te_idx = next(sgkf.split(X, strata, groups=grp))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = ytr_t[tr_idx], y[te_idx]

    # ---- CROSS-VALIDATION on the training part (stratified, grouped) --------
    oof = {k: np.zeros(len(Xtr)) for k in WEIGHTS}
    fold_r2 = []
    strata_tr = pd.qcut(y[tr_idx], N_STRATA, labels=False, duplicates="drop")
    grp_tr = grp[tr_idx]
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM)
    for f,(a,b) in enumerate(cv.split(Xtr, strata_tr, groups=grp_tr)):
        mdls = make_models()
        fp = {}
        for nm,m in mdls.items():
            m.fit(Xtr.iloc[a], ytr[a]); fp[nm] = m.predict(Xtr.iloc[b]); oof[nm][b] = fp[nm]
        pb = blend(fp)
        yb_true = y[tr_idx][b]
        pb_orig = np.expm1(pb) if log_target else pb
        fold_r2.append(r2_score(yb_true, pb_orig))
    oof_blend = blend(oof)
    oof_orig  = np.expm1(oof_blend) if log_target else oof_blend
    print("\n  --- OUT-OF-FOLD (stratified GroupKFold CV) ---")
    for nm in WEIGHTS:
        o = np.expm1(oof[nm]) if log_target else oof[nm]
        m = metrics(y[tr_idx], o); print(f"    {nm:11s} R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")
    mo = metrics(y[tr_idx], oof_orig)
    print(f"    {'BLEND':11s} R2={mo['R2']:.3f} RMSE={mo['RMSE']:.3f} MAE={mo['MAE']:.3f}")
    print(f"    fold R2: " + ", ".join(f"{r:.3f}" for r in fold_r2) +
          f"   (mean {np.mean(fold_r2):.3f} +/- {np.std(fold_r2):.3f})")

    # ---- FINAL fit on all training, evaluate locked test -------------------
    final = make_models(); tp = {}
    for nm,m in final.items():
        m.fit(Xtr, ytr); tp[nm] = m.predict(Xte)
    test_pred = blend(tp); test_pred = np.expm1(test_pred) if log_target else test_pred
    mt = metrics(yte, test_pred)
    print("\n  --- LOCKED TEST (held-out exact mixes) ---")
    print(f"    BLEND  R2={mt['R2']:.3f}  RMSE={mt['RMSE']:.3f}  MAE={mt['MAE']:.3f}")

    # ---- permutation importance (on test, blended via LightGBM proxy) ------
    imp = permutation_importance(final["LightGBM"], Xte,
                                 (np.log1p(yte) if log_target else yte),
                                 n_repeats=10, random_state=RANDOM, n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=feat_names).sort_values(ascending=False)

    # ---- SHAP (TreeExplainer on the LightGBM member of the blend) -----------
    explainer   = shap.TreeExplainer(final["LightGBM"])
    shap_values = explainer.shap_values(Xte)               # (n_test, n_features)
    shap_mean   = pd.Series(np.abs(shap_values).mean(0),
                            index=feat_names).sort_values(ascending=False)
    print("\n  --- SHAP top-10 (mean |SHAP|) ---")
    for k, v in shap_mean.head(10).items():
        print(f"    {k:20s} {v:.4f}")

    return dict(name=name, y_test=yte, p_test=test_pred, oof_true=y[tr_idx],
                oof_pred=oof_orig, fold_r2=fold_r2, imp=imp_s, mt=mt, mo=mo,
                shap_values=shap_values, X_shap=Xte, shap_mean=shap_mean,
                feat_names=feat_names)

# ------------------------------------------------------------------ plotting
def plot_target(R):
    name=R["name"]
    fig, ax = plt.subplots(2,3, figsize=(16,9)); fig.suptitle(
        f"{name}  |  Test R2={R['mt']['R2']:.3f}  RMSE={R['mt']['RMSE']:.3f}",
        fontsize=15, fontweight="bold")
    # 1 pred vs actual (test)
    a=ax[0,0]; a.scatter(R["y_test"],R["p_test"],alpha=.6,edgecolor="k",lw=.3,color="#2b7bba")
    lo,hi=min(R["y_test"].min(),R["p_test"].min()),max(R["y_test"].max(),R["p_test"].max())
    a.plot([lo,hi],[lo,hi],"r--",lw=1.5); a.set_title("Predicted vs Actual (locked test)")
    a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    # 2 OOF pred vs actual
    a=ax[0,1]; a.scatter(R["oof_true"],R["oof_pred"],alpha=.35,s=14,color="#5aae61")
    lo,hi=R["oof_true"].min(),R["oof_true"].max(); a.plot([lo,hi],[lo,hi],"r--",lw=1.5)
    a.set_title("Out-of-fold CV predictions"); a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    # 3 residuals
    res=R["p_test"]-R["y_test"]; a=ax[0,2]
    a.scatter(R["p_test"],res,alpha=.6,color="#d6604d",edgecolor="k",lw=.3)
    a.axhline(0,color="k",lw=1); a.set_title("Residuals vs Predicted")
    a.set_xlabel("Predicted"); a.set_ylabel("Residual")
    # 4 residual hist
    a=ax[1,0]; a.hist(res,bins=25,color="#9970ab",edgecolor="k")
    a.set_title(f"Residual distribution (mean={res.mean():.2f})"); a.set_xlabel("Residual")
    # 5 fold stability
    a=ax[1,1]; a.bar(range(1,len(R["fold_r2"])+1),R["fold_r2"],color="#4393c3",edgecolor="k")
    a.axhline(np.mean(R["fold_r2"]),color="r",ls="--",label=f"mean {np.mean(R['fold_r2']):.3f}")
    a.set_ylim(0,1); a.set_title("CV fold R2 (stability)"); a.set_xlabel("fold"); a.legend()
    # 6 top importances
    a=ax[1,2]; top=R["imp"].head(15)[::-1]
    a.barh(top.index,top.values,color="#f4a582",edgecolor="k")
    a.set_title("Top 15 permutation importances")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f"{OUTDIR}/{name.replace(' ','_')}_results.png",dpi=120)
    return fig

def plot_shap(R):
    """dedicated SHAP figure: beeswarm (direction) + mean|SHAP| bar (magnitude)"""
    name = R["name"]
    fig = plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}  |  SHAP explanations (locked-test set)",
                 fontsize=14, fontweight="bold")
    # left: beeswarm  -> shows feature effect direction & spread
    ax1 = fig.add_subplot(1,2,1)
    plt.sca(ax1)
    shap.summary_plot(R["shap_values"], R["X_shap"], plot_type="dot",
                      max_display=15, show=False, plot_size=None)
    ax1.set_title("Beeswarm (impact & direction)")
    # right: mean |SHAP| bar -> global importance ranking
    ax2 = fig.add_subplot(1,2,2)
    top = R["shap_mean"].head(15)[::-1]
    ax2.barh(top.index, top.values, color="#3182bd", edgecolor="k")
    ax2.set_title("Mean |SHAP| (global importance)")
    ax2.set_xlabel("mean |SHAP value|")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name.replace(' ','_')}_SHAP.png", dpi=120,
                bbox_inches="tight")
    return fig

# ------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    results = []
    results.append(run_target("RUTTING (LWT)", "Rutting_Design", "LWT_Design_Result",
                              log_target=True,  ylo=0.5, yhi=10.0))
    results.append(run_target("SCB", "SCB_Design", "SCB_Result",
                              log_target=False, ylo=0.30, yhi=1.60))
    figs  = [plot_target(R) for R in results]
    shaps = [plot_shap(R)   for R in results]
    if not HEADLESS:
        plt.show()
    print("\nDONE - result + SHAP plots saved as PNG and shown.")
