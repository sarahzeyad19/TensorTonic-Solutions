# -*- coding: utf-8 -*-
"""
CV-STRATEGY BENCHMARK for rutting (LWT @ 20k) and SCB  --  regression version
================================================================================
Adapted from the ACLR hop-trial study design:
    "623 hop trials from 72 individuals ... four CV strategies compared:
     stratified 10-fold, leave-one-participant-out (LOPOCV), group 3-fold,
     and a nested LOPOCV(outer)+group-3-fold(inner) framework; ten supervised
     models benchmarked on accuracy, train-test gap, ranking consistency, and
     computational efficiency."

Mapping to this dataset
    hop trial      -> one replicate JMF row      (replicates are KEPT, not averaged)
    individual     -> Mix_ID                      (the grouping unit / "participant")
    accuracy       -> R^2  (regression)
    4 CV strategies:
        1. Stratified 10-fold   (target-binned, IGNORES groups -> optimistic)
        2. LOPOCV               (leave-one-Mix_ID-out, pooled OOF, MC-capped)
        3. Group 3-fold         (GroupKFold by Mix_ID)
        4. Nested               (LOPOCV outer + Group 3-fold inner model selection)
    10 regressors: Ridge, ElasticNet, KNN, SVR, RandomForest, ExtraTrees,
                   GradientBoosting, XGBoost, LightGBM, CatBoost

LOPOCV note: a single Mix_ID has only ~2-3 rows, so per-fold R^2 is undefined.
We therefore POOL the out-of-group predictions across held-out mixes and compute
one R^2 (the standard LOPOCV-regression estimate). Group counts are Monte-Carlo
capped for runtime; set the *_SAMPLE knobs to None for exhaustive runs.

Run in Spyder -> prints result tables and pops up the comparison figures.
Requirements: pandas numpy scikit-learn lightgbm xgboost catboost matplotlib scipy openpyxl
"""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor

# ============================== CONFIG =======================================
FILE          = "Book1._rutting__scb_design_validation_xlsx.xlsx"
RANDOM        = 42
N_STRAT_FOLDS = 10       # stratified k-fold
N_GROUP_FOLDS = 3        # group k-fold
LOPOCV_SAMPLE = 150      # #mixes left out (pooled). None = every mix (slow)
NESTED_OUTER  = 100      # #outer LOPOCV folds for the nested framework. None = all
INNER_FOLDS   = 3        # inner group folds for nested model selection
FAST          = False    # True -> subsample + tiny models (quick smoke test)
HEADLESS      = False     # Spyder: keep False so plots show

# ---------------------------------------------------------------- file finder
def resolve_file(name):
    if os.path.isfile(name): return name
    base=os.path.basename(name); here=os.getcwd()
    for root in [here, os.path.expanduser("~/Downloads"), os.path.expanduser("~"), "C:/Users"]:
        for pat in [base, "*rutting*scb*.xlsx"]:
            hit=glob.glob(os.path.join(root,"**",pat),recursive=True)
            if hit: print(f"[info] using workbook: {hit[0]}"); return hit[0]
    raise FileNotFoundError(f"Set FILE to the full path of '{base}'.")

# ============================== FEATURES =====================================
NMAS_MAP={"1/2 in.":12.5,"3/4 in.":19.0,"1 in.":25.0,"3/8 in.":9.5}
SA_FACTOR={"Pass_No_4":0.41,"Pass_No_8":0.82,"Pass_No_16":1.64,"Pass_No_30":2.87,
           "Pass_No_50":6.14,"Pass_No_100":12.29,"Pass_No_200":32.77}
num=lambda s: pd.to_numeric(s,errors="coerce")
def pick(df,*c):
    for x in c:
        if x in df.columns: return num(df[x])
    return pd.Series(np.nan,index=df.index)
def build_features(df):
    P,A="Design_Submission__","Combined_Aggregate_"; X=pd.DataFrame(index=df.index)
    X["Va"]=pick(df,P+"Percent_Voids");X["VMA"]=pick(df,P+"VMA");X["VFA"]=pick(df,P+"VFA")
    X["Pbe"]=pick(df,P+"Pbe");X["Pba"]=pick(df,P+"Pba");X["Gse"]=pick(df,P+"Gse")
    X["Gmm"]=pick(df,P+"Gmm");X["AC"]=pick(df,P+"Percent_AC");X["Dust_Pbe"]=pick(df,P+"Dust_Pbeff")
    X["Gmm_Ni"]=pick(df,P+"Percent_Gmm_Ni");X["Gmm_Nm"]=pick(df,P+"Percent_Gmm_Nm")
    X["Absorption"]=pick(df,A+"Absorption");X["FAA"]=pick(df,A+"FAA");X["CAA"]=pick(df,A+"CAA")
    X["SandEq"]=pick(df,A+"Sand_Equivalent");X["FlatElong"]=pick(df,A+"Flat_Elongated");X["Gsb"]=pick(df,A+"Bulk_Gravity")
    for s in ["Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4","Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]:
        X[s]=pick(df,P+s)
    X["ADT"]=pick(df,"ADT");X["MixTemp"]=pick(df,"Mix_Temperature");X["AC_from_RAP"]=pick(df,"Total_AC_From_RAP")
    X["NMAS_mm"]=df["Nominal_Aggregate_Size"].map(NMAS_MAP)
    for c in ["Design_Level","Mix_Type"]:
        d=pd.get_dummies(df[c].astype(str),prefix=c).astype(float); X=pd.concat([X,d.set_index(X.index)],axis=1)
    X["RAP_Binder_Ratio"]=X["AC_from_RAP"]/X["AC"].replace(0,np.nan)
    X["VirginBinder"]=(X["AC"]-X["AC_from_RAP"]).clip(lower=0)
    X["Fines_to_Pbe"]=X["Pass_No_200"]/X["Pbe"].replace(0,np.nan)
    X["Coarse_Fraction"]=100-X["Pass_No_4"];X["Intermediate_Frac"]=X["Pass_No_4"]-X["Pass_No_8"];X["Fine_Fraction"]=X["Pass_No_8"]-X["Pass_No_200"]
    X["VMA_Filled_Index"]=X["VMA"]*X["VFA"]/100;X["AbsorptionPenalty"]=X["Absorption"]*X["AC"]
    X["PostDens"]=X["Gmm_Nm"]-96;X["InitialDensity"]=X["Gmm_Ni"]
    SA=sum(X[k]*v for k,v in SA_FACTOR.items())/100+0.41;X["SurfaceArea"]=SA
    X["AFT_micron"]=(X["Pbe"]/100)/(SA*X["Gsb"].replace(0,np.nan))*1000
    X["CrackingExposure"]=X["Va"]*SA/X["Pbe"].replace(0,np.nan);X["PG_proxy_x_Va"]=X["MixTemp"]*X["Va"]
    return X
def load_target(path,sheet,tgt,ylo,yhi):
    df=pd.read_excel(path,sheet_name=sheet)
    y=num(df[tgt]); X=build_features(df)
    keep=np.isfinite(y)&(y>=ylo)&(y<=yhi)
    X=X[keep].reset_index(drop=True); y=y[keep].values
    grp=np.array([str(v) for v in df.loc[keep,"Mix_ID"].tolist()], dtype=object)  # KEEP replicates; group=Mix_ID
    X=X.replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if FAST:                                               # smoke-test subsample
        rng=np.random.RandomState(RANDOM); idx=rng.choice(len(X),min(700,len(X)),replace=False)
        X,y,grp=X.iloc[idx].reset_index(drop=True),y[idx],grp[idx]
    return X,y,grp

# ============================== MODELS (10) ==================================
def model_zoo():
    T = 120 if FAST else 300
    I = 150 if FAST else 300
    return {
        "Ridge":        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM)),
        "ElasticNet":   lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=RANDOM)),
        "KNN":          lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "SVR":          lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=T, n_jobs=-1, random_state=RANDOM),
        "ExtraTrees":   lambda: ExtraTreesRegressor(n_estimators=T, n_jobs=-1, random_state=RANDOM),
        "GradBoost":    lambda: GradientBoostingRegressor(n_estimators=I, learning_rate=0.05, max_depth=3, random_state=RANDOM),
        "XGBoost":      lambda: xgb.XGBRegressor(n_estimators=I, learning_rate=0.05, max_depth=5, subsample=0.85,
                                    colsample_bytree=0.8, random_state=RANDOM, n_jobs=-1, verbosity=0),
        "LightGBM":     lambda: lgb.LGBMRegressor(n_estimators=I, learning_rate=0.05, num_leaves=31,
                                    subsample=0.85, colsample_bytree=0.8, random_state=RANDOM, n_jobs=-1, verbose=-1),
        "CatBoost":     lambda: CatBoostRegressor(iterations=I, learning_rate=0.05, depth=6,
                                    random_state=RANDOM, verbose=0),
    }

# ============================== CV ENGINES ===================================
def pooled_oof(make, X, y, splits):
    """fit on each train split, predict held-out; pool test preds -> one R2.
       returns pooled_test_R2, mean_train_R2, total_fit_time, n_test_pred"""
    yhat=np.full(len(y),np.nan); tr_r2=[]; t=0.0
    for tr,te in splits:
        m=make(); t0=time.time(); m.fit(X.iloc[tr],y[tr]); t+=time.time()-t0
        yhat[te]=m.predict(X.iloc[te]); tr_r2.append(r2_score(y[tr],m.predict(X.iloc[tr])))
    mask=~np.isnan(yhat)
    return r2_score(y[mask],yhat[mask]), float(np.mean(tr_r2)), t, int(mask.sum())

def make_splits(strategy, X, y, grp, rng):
    ybin=pd.qcut(y, min(N_STRAT_FOLDS,10), labels=False, duplicates="drop")
    if strategy=="Stratified10":
        return list(StratifiedKFold(N_STRAT_FOLDS, shuffle=True, random_state=RANDOM).split(X, ybin))
    if strategy=="GroupKFold3":
        return list(GroupKFold(N_GROUP_FOLDS).split(X, y, groups=grp))
    if strategy=="LOPOCV":
        splits=list(LeaveOneGroupOut().split(X, y, groups=grp))
        if LOPOCV_SAMPLE and LOPOCV_SAMPLE < len(splits):
            sel=rng.choice(len(splits), LOPOCV_SAMPLE, replace=False); splits=[splits[i] for i in sel]
        return splits
    raise ValueError(strategy)

def run_nested(zoo, X, y, grp, rng):
    """LOPOCV outer + GroupKFold(inner) model selection. Pooled outer R2 +
       selection frequency of each model."""
    outer=list(LeaveOneGroupOut().split(X,y,groups=grp))
    if NESTED_OUTER and NESTED_OUTER < len(outer):
        sel=rng.choice(len(outer), NESTED_OUTER, replace=False); outer=[outer[i] for i in sel]
    yhat=np.full(len(y),np.nan); picks={k:0 for k in zoo}; t0=time.time()
    for tr,te in outer:
        gtr=grp[tr]
        inner=list(GroupKFold(min(INNER_FOLDS,len(np.unique(gtr)))).split(X.iloc[tr],y[tr],groups=gtr))
        best=(-9,None)
        for nm,make in zoo.items():
            r2,_,_,_=pooled_oof(make, X.iloc[tr].reset_index(drop=True), y[tr], inner)
            if r2>best[0]: best=(r2,nm)
        picks[best[1]]+=1
        m=zoo[best[1]](); m.fit(X.iloc[tr],y[tr]); yhat[te]=m.predict(X.iloc[te])
    mask=~np.isnan(yhat)
    return r2_score(y[mask],yhat[mask]), time.time()-t0, picks

# ============================== DRIVER =======================================
def benchmark(path, name, sheet, tgt, ylo, yhi):
    print("\n"+"#"*74+f"\n#  {name}\n"+"#"*74)
    X,y,grp=load_target(path,sheet,tgt,ylo,yhi)
    print(f"  rows (trials)={len(X)}   mixes (individuals)={len(np.unique(grp))}   "
          f"features={X.shape[1]}   target range=[{y.min():.2f},{y.max():.2f}]")
    zoo=model_zoo(); rng=np.random.RandomState(RANDOM)
    strategies=["Stratified10","LOPOCV","GroupKFold3"]
    rows=[]
    for strat in strategies:
        splits=make_splits(strat, X, y, grp, rng)
        print(f"\n  [{strat}]  ({len(splits)} folds)")
        for nm,make in zoo.items():
            r2,tr,t,n=pooled_oof(make, X, y, splits)
            rows.append(dict(strategy=strat, model=nm, test_R2=r2, train_R2=tr,
                             gap=tr-r2, time_s=t, n_pred=n))
            print(f"     {nm:13s} R2={r2:5.3f}  gap={tr-r2:5.3f}  time={t:6.2f}s")
    res=pd.DataFrame(rows)
    # nested framework
    print("\n  [Nested: LOPOCV outer + GroupKFold inner]")
    nr2, nt, picks=run_nested(zoo, X, y, grp, rng)
    tot=sum(picks.values()); freq={k:round(v/tot,3) for k,v in picks.items() if v>0}
    print(f"     Nested pooled R2={nr2:.3f}   time={nt:.1f}s")
    print(f"     model selection frequency: {freq}")
    return dict(name=name, res=res, nested_r2=nr2, nested_time=nt, picks=picks)

# ============================== ANALYSIS / PLOTS =============================
STRAT_ORDER = ["Stratified10","LOPOCV","GroupKFold3"]
STRAT_COL   = {"Stratified10":"#d6604d","LOPOCV":"#4393c3","GroupKFold3":"#5aae61"}

def plot_benchmark(B):
    name=B["name"]; res=B["res"]
    piv   = res.pivot(index="model", columns="strategy", values="test_R2")[STRAT_ORDER]
    gappv = res.pivot(index="model", columns="strategy", values="gap")[STRAT_ORDER]
    timepv= res.pivot(index="model", columns="strategy", values="time_s")[STRAT_ORDER]
    order = piv["GroupKFold3"].sort_values(ascending=False).index.tolist()
    piv,gappv,timepv = piv.loc[order],gappv.loc[order],timepv.loc[order]
    models=order; xpos=np.arange(len(models)); w=0.26

    fig,ax=plt.subplots(2,2,figsize=(16,10))
    fig.suptitle(f"{name}  |  4-CV-strategy x 10-model benchmark",fontsize=15,fontweight="bold")
    # (1) test R2 per model x strategy
    a=ax[0,0]
    for i,s in enumerate(STRAT_ORDER):
        a.bar(xpos+(i-1)*w, piv[s].values, w, label=s, color=STRAT_COL[s], edgecolor="k",lw=.4)
    a.axhline(B["nested_r2"],color="purple",ls="--",lw=1.6,label=f"Nested pooled R2={B['nested_r2']:.3f}")
    a.set_xticks(xpos); a.set_xticklabels(models,rotation=45,ha="right"); a.set_ylabel("pooled test R2")
    a.set_title("Accuracy (R2) by model & CV strategy"); a.legend(fontsize=8)
    # (2) generalization gap
    a=ax[0,1]
    for i,s in enumerate(STRAT_ORDER):
        a.bar(xpos+(i-1)*w, gappv[s].values, w, label=s, color=STRAT_COL[s], edgecolor="k",lw=.4)
    a.set_xticks(xpos); a.set_xticklabels(models,rotation=45,ha="right")
    a.set_ylabel("train R2  -  test R2"); a.set_title("Train-test generalization gap"); a.legend(fontsize=8)
    # (3) ranking-consistency heatmap (Spearman between strategies' model rankings)
    a=ax[1,0]; S=len(STRAT_ORDER); M=np.ones((S,S))
    for i in range(S):
        for j in range(S):
            M[i,j]=spearmanr(piv[STRAT_ORDER[i]], piv[STRAT_ORDER[j]]).correlation
    im=a.imshow(M,vmin=0,vmax=1,cmap="viridis")
    a.set_xticks(range(S)); a.set_xticklabels(STRAT_ORDER,rotation=30,ha="right")
    a.set_yticks(range(S)); a.set_yticklabels(STRAT_ORDER)
    for i in range(S):
        for j in range(S): a.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",
                                  color="w" if M[i,j]<0.7 else "k",fontsize=10)
    a.set_title("Model-ranking consistency (Spearman)"); fig.colorbar(im,ax=a,fraction=0.046)
    # (4) computational efficiency (total fit time, log scale)
    a=ax[1,1]
    for i,s in enumerate(STRAT_ORDER):
        a.bar(xpos+(i-1)*w, timepv[s].values, w, label=s, color=STRAT_COL[s], edgecolor="k",lw=.4)
    a.set_yscale("log"); a.set_xticks(xpos); a.set_xticklabels(models,rotation=45,ha="right")
    a.set_ylabel("total fit time (s, log)"); a.set_title("Computational efficiency"); a.legend(fontsize=8)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f"{OUTDIR}/{name.split('(')[0].strip().replace(' ','_')}_CVbenchmark.png",dpi=120)

    # (5) model-selection frequency of the nested framework
    fig2,ax2=plt.subplots(figsize=(9,4.5))
    picks=pd.Series(B["picks"]); picks=picks[picks>0].sort_values()
    ax2.barh(picks.index, picks.values, color="#9970ab", edgecolor="k")
    ax2.set_xlabel("times selected by inner GroupKFold (outer folds)")
    ax2.set_title(f"{name}  |  Nested framework - model selection frequency")
    fig2.tight_layout()
    fig2.savefig(f"{OUTDIR}/{name.split('(')[0].strip().replace(' ','_')}_nested_selection.png",dpi=120)
    return fig,fig2

def summary_table(B):
    res=B["res"]; piv=res.pivot(index="model",columns="strategy",values="test_R2")[STRAT_ORDER]
    piv=piv.loc[piv["GroupKFold3"].sort_values(ascending=False).index]
    print(f"\n===== {B['name']}  pooled test R2 =====")
    print(piv.round(3).to_string())
    # optimism = stratified(leaky) - grouped(honest)
    opt=(piv["Stratified10"]-piv["GroupKFold3"])
    print(f"  mean optimism (Stratified10 - GroupKFold3): {opt.mean():.3f}")
    cons=[spearmanr(piv[a],piv[b]).correlation for a in STRAT_ORDER for b in STRAT_ORDER if a<b]
    print(f"  mean ranking consistency (Spearman): {np.mean(cons):.3f}")

OUTDIR="."
# ============================== MAIN =========================================
if __name__ == "__main__":
    print("[note] full run with defaults (LOPOCV_SAMPLE=150, NESTED_OUTER=100) "
          "takes ~20-40 min. Set FAST=True for a quick smoke test.")
    path=resolve_file(FILE)
    Bs=[]
    Bs.append(benchmark(path,"RUTTING (LWT @ 20k passes)","Rutting_Design","LWT_Design_Result",0.5,10.0))
    Bs.append(benchmark(path,"SCB","SCB_Design","SCB_Result",0.30,1.60))
    for B in Bs: summary_table(B)
    for B in Bs: plot_benchmark(B)
    if not HEADLESS: plt.show()
    print("\nDONE - benchmark tables printed and figures saved/shown.")

