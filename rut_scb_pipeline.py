# -*- coding: utf-8 -*-
"""
Rutting (LWT @ 20,000 passes) and SCB prediction  -  strong, leakage-safe pipeline
================================================================================
Method (as requested):
    * target = LWT rut depth at 20,000 passes  (LWT_Design_Result)  and  SCB
    * build engineering feature set from the JMF design submission
    * remove duplicate EXACT mixes  ->  one row per Exact_JMF_Group
      (group key = Mix_ID + full rounded composition ; NO family-level grouping)
    * STRATIFIED, GROUP-aware splitting (replaces random split):
         - StratifiedGroupKFold holdout  (stratified on target bins, grouped by exact mix)
         - StratifiedGroupKFold cross-validation
    * SHAP-based feature selection : keep the TOP_K features by mean|SHAP|,
      drop the ones that do not develop the model (selection uses TRAIN only)
    * tuned ExtraTrees / LightGBM / CatBoost / XGBoost with OOF-optimized blend weights
    * log target for rutting
Run directly in Spyder -> prints metrics + SHAP ranking and pops up all plots.

Requirements:
    pip install pandas numpy scikit-learn lightgbm catboost xgboost shap matplotlib openpyxl
"""
import os, glob, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap

# ============================== CONFIG =======================================
# 1) put the workbook name/path here. If the exact path is not found the script
#    automatically searches the script folder and your Downloads folder for it.
FILE       = "Book1._rutting__scb_design_validation_xlsx.xlsx"
TOP_K      = 20        # keep this many SHAP-top features (None = keep all)
N_SPLITS   = 5
N_STRATA   = 5         # target quantile bins for stratification
RANDOM     = 42
OUTDIR     = "."
HEADLESS   = False     # Spyder: keep False so plots pop up on screen

# ---------------------------------------------------------------- file finder
def resolve_file(name):
    if os.path.isfile(name):
        return name
    base = os.path.basename(name)
    cands = []
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for root in [here, os.getcwd(), os.path.expanduser("~/Downloads"),
                 os.path.expanduser("~"), "C:/Users"]:
        cands += glob.glob(os.path.join(root, "**", base), recursive=True)
        cands += glob.glob(os.path.join(root, "**", "*rutting*scb*.xlsx"), recursive=True)
    cands = [c for c in cands if os.path.isfile(c)]
    if cands:
        print(f"[info] using workbook: {cands[0]}")
        return cands[0]
    raise FileNotFoundError(
        f"Could not find '{base}'. Set FILE at the top of the script to the full path, e.g.\n"
        r"   FILE = r'C:/Users/lenovo/Downloads/Book1._rutting__scb_design_validation_xlsx.xlsx'")

# ============================== FEATURES =====================================
NMAS_MAP  = {"1/2 in.":12.5, "3/4 in.":19.0, "1 in.":25.0, "3/8 in.":9.5}
SA_FACTOR = {"Pass_No_4":0.41,"Pass_No_8":0.82,"Pass_No_16":1.64,"Pass_No_30":2.87,
             "Pass_No_50":6.14,"Pass_No_100":12.29,"Pass_No_200":32.77}
num = lambda s: pd.to_numeric(s, errors="coerce")

def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return num(df[c])
    return pd.Series(np.nan, index=df.index)

def build_features(df):
    P, A = "Design_Submission__", "Combined_Aggregate_"
    X = pd.DataFrame(index=df.index)
    # volumetrics
    X["Va"]=pick(df,P+"Percent_Voids"); X["VMA"]=pick(df,P+"VMA"); X["VFA"]=pick(df,P+"VFA")
    X["Pbe"]=pick(df,P+"Pbe"); X["Pba"]=pick(df,P+"Pba"); X["Gse"]=pick(df,P+"Gse")
    X["Gmm"]=pick(df,P+"Gmm"); X["AC"]=pick(df,P+"Percent_AC"); X["Dust_Pbe"]=pick(df,P+"Dust_Pbeff")
    X["Gmm_Ni"]=pick(df,P+"Percent_Gmm_Ni"); X["Gmm_Nm"]=pick(df,P+"Percent_Gmm_Nm")
    # aggregate
    X["Absorption"]=pick(df,A+"Absorption"); X["FAA"]=pick(df,A+"FAA"); X["CAA"]=pick(df,A+"CAA")
    X["SandEq"]=pick(df,A+"Sand_Equivalent"); X["FlatElong"]=pick(df,A+"Flat_Elongated"); X["Gsb"]=pick(df,A+"Bulk_Gravity")
    # gradation
    for s in ["Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4","Pass_No_8",
              "Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]:
        X[s]=pick(df,P+s)
    # traffic / design
    X["ADT"]=pick(df,"ADT"); X["MixTemp"]=pick(df,"Mix_Temperature"); X["AC_from_RAP"]=pick(df,"Total_AC_From_RAP")
    X["NMAS_mm"]=df["Nominal_Aggregate_Size"].map(NMAS_MAP)
    # categoricals -> one-hot (nominal levels; better than arbitrary ordinal codes)
    for c in ["Design_Level","Mix_Type"]:
        d = pd.get_dummies(df[c].astype(str), prefix=c).astype(float)
        X = pd.concat([X, d.set_index(X.index)], axis=1)
    # ---------------- ENGINEERING FEATURES ----------------
    X["RAP_Binder_Ratio"]  = X["AC_from_RAP"]/X["AC"].replace(0,np.nan)     # RBR proxy
    X["VirginBinder"]      = (X["AC"]-X["AC_from_RAP"]).clip(lower=0)
    X["Fines_to_Pbe"]      = X["Pass_No_200"]/X["Pbe"].replace(0,np.nan)
    X["Coarse_Fraction"]   = 100-X["Pass_No_4"]
    X["Intermediate_Frac"] = X["Pass_No_4"]-X["Pass_No_8"]
    X["Fine_Fraction"]     = X["Pass_No_8"]-X["Pass_No_200"]
    X["VMA_Filled_Index"]  = X["VMA"]*X["VFA"]/100.0
    X["AbsorptionPenalty"] = X["Absorption"]*X["AC"]
    X["PostDensification"] = X["Gmm_Nm"]-96.0
    X["InitialDensity"]    = X["Gmm_Ni"]
    SA = sum(X[k]*v for k,v in SA_FACTOR.items())/100.0 + 0.41
    X["SurfaceArea"]       = SA
    X["AFT_micron"]        = (X["Pbe"]/100.0)/(SA*X["Gsb"].replace(0,np.nan))*1000.0
    X["CrackingExposure"]  = X["Va"]*SA/X["Pbe"].replace(0,np.nan)
    X["PG_proxy_x_Va"]     = X["MixTemp"]*X["Va"]
    return X

def make_dataset(path, sheet, target_col, ylo, yhi):
    df = pd.read_excel(path, sheet_name=sheet)
    y  = num(df[target_col])
    ident = [c for c in ["Mix_ID","Design_Level"] if c in df.columns]
    sieves = ["Pass_1_1_2in","Pass_1in","Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4",
              "Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]
    comp = ["Design_Submission__Percent_AC","Design_Submission__Percent_Voids",
            "Design_Submission__VMA","Design_Submission__VFA","Design_Submission__Pbe",
            "Design_Submission__Gse"] + ["Design_Submission__"+s for s in sieves] + \
           ["Nominal_Aggregate_Size","Mix_Type","Total_AC_From_RAP"]
    keycols = [c for c in ident+comp if c in df.columns]
    ks = df[keycols].copy()
    for c in keycols:
        if ks[c].dtype.kind in "fc":
            ks[c] = num(ks[c]).round(2)
    exact = ks.apply(lambda col: col.map(str)).agg("|".join, axis=1)
    X = build_features(df); X["__y__"]=y; X["__grp__"]=exact.values
    X = X[np.isfinite(X["__y__"]) & (X["__y__"]>=ylo) & (X["__y__"]<=yhi)]   # winsorize errors
    agg = X.groupby("__grp__").mean(numeric_only=True)                        # one row / exact mix
    rep = X.groupby("__grp__").size().values
    agg = agg.reset_index()
    y   = agg.pop("__y__").values
    grp = agg.pop("__grp__").values
    agg = agg.fillna(agg.median())
    return agg, y, grp, rep

# ============================== MODELS =======================================
def make_models():
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=1200, min_samples_leaf=1,
                        max_features=0.6, random_state=RANDOM, n_jobs=-1),
        "LightGBM":   lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.015, num_leaves=63,
                        min_child_samples=15, subsample=0.85, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=1.0, random_state=RANDOM,
                        n_jobs=-1, verbose=-1),
        "CatBoost":   CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=8,
                        l2_leaf_reg=3.0, random_state=RANDOM, verbose=0),
        "XGBoost":    xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=6,
                        subsample=0.85, colsample_bytree=0.7, reg_lambda=1.5,
                        min_child_weight=3, random_state=RANDOM, n_jobs=-1, verbosity=0),
    }
def metrics(y,p):
    return dict(R2=r2_score(y,p), RMSE=np.sqrt(mean_squared_error(y,p)), MAE=mean_absolute_error(y,p))
def optimize_weights(oof, y_true, inv):
    keys=list(oof); best=(-9,None)
    for w in itertools.product(np.arange(0,1.01,0.1), repeat=len(keys)):
        if abs(sum(w)-1)>1e-6: continue
        p=sum(w[i]*oof[keys[i]] for i in range(len(keys)))
        s=r2_score(y_true, inv(p))
        if s>best[0]: best=(s, dict(zip(keys,w)))
    return best

# ============================== CORE RUN =====================================
def run_target(path, name, sheet, target_col, log_target, ylo, yhi):
    print("\n"+"="*70+f"\n  {name}\n"+"="*70)
    X, y, grp, rep = make_dataset(path, sheet, target_col, ylo, yhi)
    feat_names = list(X.columns)
    inv = (lambda p: np.expm1(p)) if log_target else (lambda p: p)
    yt  = np.log1p(y) if log_target else y
    print(f"  exact-mix groups (rows after dedup): {len(X)}   (avg replicates merged: {rep.mean():.2f})")
    print(f"  target mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.2f},{y.max():.2f}]")

    strata = pd.qcut(y, N_STRATA, labels=False, duplicates="drop")
    sgkf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM)
    tr, te = next(sgkf.split(X, strata, groups=grp))
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    ytr, yte = yt[tr], y[te]

    # ---- SHAP feature selection (TRAIN only, no leakage) -------------------
    if TOP_K and TOP_K < len(feat_names):
        prelim = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                   random_state=RANDOM, verbose=-1).fit(Xtr, ytr)
        sv = shap.TreeExplainer(prelim).shap_values(Xtr)
        rank = pd.Series(np.abs(sv).mean(0), index=feat_names).sort_values(ascending=False)
        keep = list(rank.head(TOP_K).index)
        dropped = [f for f in feat_names if f not in keep]
        Xtr, Xte, feat_names = Xtr[keep], Xte[keep], keep
        print(f"  SHAP feature selection: kept {len(keep)} / dropped {len(dropped)}")
        print(f"    dropped: {', '.join(dropped[:12])}{' ...' if len(dropped)>12 else ''}")

    # ---- CV on training (stratified, grouped) -> OOF ----------------------
    strata_tr = pd.qcut(y[tr], N_STRATA, labels=False, duplicates="drop"); grp_tr = grp[tr]
    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM)
    oof = {k: np.zeros(len(Xtr)) for k in make_models()}
    fold_r2 = []
    for a,b in cv.split(Xtr, strata_tr, groups=grp_tr):
        fp={}
        for nm,m in make_models().items():
            m.fit(Xtr.iloc[a], ytr[a]); fp[nm]=m.predict(Xtr.iloc[b]); oof[nm][b]=fp[nm]
        fold_r2.append(r2_score(y[tr][b], inv(np.mean(list(fp.values()),axis=0))))
    print("\n  --- OUT-OF-FOLD (stratified GroupKFold CV) ---")
    for nm in oof:
        m=metrics(y[tr], inv(oof[nm])); print(f"    {nm:11s} R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")
    best_r2, W = optimize_weights(oof, y[tr], inv)
    print(f"    optimized weights = {{{', '.join(f'{k}:{W[k]:.2f}' for k in W)}}}")
    print(f"    BLEND R2={best_r2:.3f}   fold R2: "+", ".join(f'{r:.3f}' for r in fold_r2)
          +f"  (mean {np.mean(fold_r2):.3f} +/- {np.std(fold_r2):.3f})")
    oof_blend = inv(sum(W[k]*oof[k] for k in W))

    # ---- FINAL fit on train, evaluate locked test -------------------------
    final = make_models(); tp={}
    for nm,m in final.items():
        m.fit(Xtr, ytr); tp[nm]=m.predict(Xte)
    test_pred = inv(sum(W[k]*tp[k] for k in W))
    mt = metrics(yte, test_pred)
    print("\n  --- LOCKED TEST (held-out exact mixes) ---")
    print(f"    BLEND  R2={mt['R2']:.3f}  RMSE={mt['RMSE']:.3f}  MAE={mt['MAE']:.3f}")

    # ---- SHAP on the final LightGBM member --------------------------------
    expl = shap.TreeExplainer(final["LightGBM"]); shap_values = expl.shap_values(Xte)
    shap_mean = pd.Series(np.abs(shap_values).mean(0), index=feat_names).sort_values(ascending=False)
    print("\n  --- SHAP top-10 (mean |SHAP|) ---")
    for k,v in shap_mean.head(10).items(): print(f"    {k:20s} {v:.4f}")

    return dict(name=name, y_test=yte, p_test=test_pred, oof_true=y[tr], oof_pred=oof_blend,
                fold_r2=fold_r2, mt=mt, shap_values=shap_values, X_shap=Xte,
                shap_mean=shap_mean, feat_names=feat_names)

# ============================== PLOTS ========================================
def plot_target(R):
    name=R["name"]; fig,ax=plt.subplots(2,3,figsize=(16,9))
    fig.suptitle(f"{name}  |  Test R2={R['mt']['R2']:.3f}  RMSE={R['mt']['RMSE']:.3f}",
                 fontsize=15, fontweight="bold")
    a=ax[0,0]; a.scatter(R["y_test"],R["p_test"],alpha=.6,edgecolor="k",lw=.3,color="#2b7bba")
    lo,hi=min(R["y_test"].min(),R["p_test"].min()),max(R["y_test"].max(),R["p_test"].max())
    a.plot([lo,hi],[lo,hi],"r--",lw=1.5); a.set_title("Predicted vs Actual (locked test)")
    a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    a=ax[0,1]; a.scatter(R["oof_true"],R["oof_pred"],alpha=.35,s=14,color="#5aae61")
    lo,hi=R["oof_true"].min(),R["oof_true"].max(); a.plot([lo,hi],[lo,hi],"r--",lw=1.5)
    a.set_title("Out-of-fold CV predictions"); a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    res=R["p_test"]-R["y_test"]; a=ax[0,2]
    a.scatter(R["p_test"],res,alpha=.6,color="#d6604d",edgecolor="k",lw=.3)
    a.axhline(0,color="k",lw=1); a.set_title("Residuals vs Predicted"); a.set_xlabel("Predicted"); a.set_ylabel("Residual")
    a=ax[1,0]; a.hist(res,bins=25,color="#9970ab",edgecolor="k")
    a.set_title(f"Residual distribution (mean={res.mean():.2f})"); a.set_xlabel("Residual")
    a=ax[1,1]; a.bar(range(1,len(R["fold_r2"])+1),R["fold_r2"],color="#4393c3",edgecolor="k")
    a.axhline(np.mean(R["fold_r2"]),color="r",ls="--",label=f"mean {np.mean(R['fold_r2']):.3f}")
    a.set_ylim(0,1); a.set_title("CV fold R2 (stability)"); a.set_xlabel("fold"); a.legend()
    a=ax[1,2]; top=R["shap_mean"].head(15)[::-1]
    a.barh(top.index,top.values,color="#f4a582",edgecolor="k"); a.set_title("Top 15 features (mean |SHAP|)")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{OUTDIR}/{name.split('(')[0].strip().replace(' ','_')}_results.png",dpi=120)
    return fig

def plot_shap(R):
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}  |  SHAP explanations (locked-test set)", fontsize=14, fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap_values"], R["X_shap"], plot_type="dot", max_display=15, show=False, plot_size=None)
    ax1.set_title("Beeswarm (impact & direction)")
    ax2=fig.add_subplot(1,2,2); top=R["shap_mean"].head(15)[::-1]
    ax2.barh(top.index,top.values,color="#3182bd",edgecolor="k")
    ax2.set_title("Mean |SHAP| (global importance)"); ax2.set_xlabel("mean |SHAP value|")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name.split('(')[0].strip().replace(' ','_')}_SHAP.png",dpi=120,bbox_inches="tight")
    return fig

# ============================== MAIN =========================================
if __name__ == "__main__":
    path = resolve_file(FILE)
    results = []
    results.append(run_target(path, "RUTTING (LWT @ 20k passes)", "Rutting_Design",
                              "LWT_Design_Result", log_target=True,  ylo=0.5,  yhi=10.0))
    results.append(run_target(path, "SCB", "SCB_Design",
                              "SCB_Result",        log_target=False, ylo=0.30, yhi=1.60))
    [plot_target(R) for R in results]
    [plot_shap(R)   for R in results]
    if not HEADLESS:
        plt.show()
    print("\nDONE - result + SHAP plots saved as PNG and shown.")
