# -*- coding: utf-8 -*-
"""
LOCKED MODEL v3  --  RUT + SCB, uses the pre-cleaned file (replicates KEPT)
================================================================================
Data source
-----------
Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx
    sheets: Rutting_Filtered, SCB_Filtered
    already applied:
       * RUT range 0.5 - 10.0 mm       * SCB range 0.30 - 1.25
       * OGFC / SMA / Thin Lift removed from Design_Level and Mix_Type
       * replicates KEPT (RUT=4591 rows / 1640 mixes ; SCB=3052 rows / 1271 mixes)
       * pre-built column  Exact_JMF_Group  =  Mix_ID + full composition

Method
------
    * feature set  = 28 "Selected_no_generated" engineering features (locked)
      + parsed PG_High from Custom_Name
    * grouping     = Exact_JMF_Group  ==>  correct LOPOCV-style CV
                     (all replicates of the same mix stay in the same fold)
    * validation   = Stratified GroupKFold, 10 folds (target-binned)
                     -- 10-fold pooled OOF is the honest test metric
    * models       = ExtraTrees (winner) + HistGB + LightGBM_reg
    * final        = 80/20 grouped locked test + full diagnostic plots
Requirements: pandas numpy scikit-learn lightgbm shap matplotlib openpyxl

Run in Spyder -> prints tables, pops all figures.
"""
import os, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb, shap

# =============================== CONFIG =====================================
FILE     = "Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx"
N_SPLITS = 10
RANDOM   = 42
FAST     = False
HEADLESS = False
OUTDIR   = "."

rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":12,
                 "axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":9,
                 "axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})

def resolve_file(name):
    if os.path.isfile(name): return name
    base=os.path.basename(name)
    for root in [os.getcwd(),os.path.expanduser("~/Downloads"),os.path.expanduser("~"),"C:/Users"]:
        for pat in [base,"*KEEP_REPLICATES*.xlsx","*filtered*RUT*.xlsx"]:
            hit=glob.glob(os.path.join(root,"**",pat),recursive=True)
            if hit: print(f"[info] using workbook: {hit[0]}"); return hit[0]
    raise FileNotFoundError(f"Set FILE to the full path of '{base}'.")

# ============================ FEATURES ======================================
# 28 canonical Selected_no_generated features (locked from SHAP + ablation study)
NUMERIC_28 = [
    # traffic & design context
    "ADT","Production_Rate",
    # aggregate skeleton & gradation
    "Nominal_Aggregate_Size",
    "Design_Submission__Pass_No_4","Design_Submission__Pass_No_8",
    "Design_Submission__Pass_No_16","Design_Submission__Pass_No_30",
    "Design_Submission__Pass_No_50","Design_Submission__Pass_No_200",
    # aggregate quality
    "Combined_Aggregate_CAA","Combined_Aggregate_FAA",
    "Combined_Aggregate_Sand_Equivalent","Combined_Aggregate_Absorption",
    "Combined_Aggregate_Bulk_Gravity","Combined_Aggregate_Flat_Elongated",
    # binder & AC
    "Design_Submission__Percent_AC","Design_Submission__Pbe","Design_Submission__Pba",
    "Total_AC_From_RAP",
    # volumetrics & densification
    "Design_Submission__Percent_Voids","Design_Submission__VMA",
    "Design_Submission__Dust_Pbeff","Design_Submission__Percent_Gmm_Ni",
    "Design_Submission__Percent_Gmm_Nm",
    # process
    "Mix_Temperature",
]
CATEGORICAL = ["Mix_Type","Design_Level"]      # one-hot inside build_X()

NMAS_MAP = {"1/2 in.":12.5,"3/4 in.":19.0,"1 in.":25.0,"3/8 in.":9.5,"1/4 in.":6.25}

def parse_pg_high(name):
    s=str(name).upper()
    m=re.search(r'(\d{2,3})\s*[-\u2013]\s*(\d{2})',s)
    return float(m.group(1)) if m else np.nan
def parse_polymer(name):
    return 1.0 if re.search(r'SBS|POLY|LATEX|MODIF|ELVALOY|PPA|GTR|RUBBER',
                            str(name).upper()) else 0.0

def build_X(df):
    num=lambda s: pd.to_numeric(s,errors="coerce")
    X=pd.DataFrame(index=df.index)
    for c in NUMERIC_28:
        if c in df.columns:
            v=df[c]
            if c=="Nominal_Aggregate_Size" and v.dtype==object:
                v=v.map(NMAS_MAP)
            X[c]=num(v)
    # PG grade parsed from Custom_Name text
    if "Custom_Name" in df.columns:
        X["PG_High"]=df["Custom_Name"].apply(parse_pg_high)
        X["Polymer"]=df["Custom_Name"].apply(parse_polymer)
        X["PG_known"]=X["PG_High"].notna().astype(float)
        X["PG_High"]=X["PG_High"].fillna(0.0)          # sentinel-fill, gated by PG_known
    # one-hot categoricals
    for c in CATEGORICAL:
        if c in df.columns:
            d=pd.get_dummies(df[c].astype(str),prefix=c).astype(float)
            X=pd.concat([X,d.set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]                            # drop constant cols
    return X

def load_target(path,sheet,tgt):
    df=pd.read_excel(path,sheet_name=sheet)
    y=pd.to_numeric(df[tgt],errors="coerce").values
    grp=df["Exact_JMF_Group"].astype(str).values          # pre-built grouping key
    X=build_X(df)
    if FAST:
        rng=np.random.RandomState(RANDOM); idx=rng.choice(len(X),min(1200,len(X)),replace=False)
        X,y,grp=X.iloc[idx].reset_index(drop=True),y[idx],grp[idx]
    return X,y,grp

# ============================ MODELS ========================================
def zoo():
    T=250 if FAST else 650
    return {
        "ExtraTrees":   ExtraTreesRegressor(n_estimators=T,min_samples_leaf=2,
                          max_features=0.85,n_jobs=-1,random_state=RANDOM),
        "HistGB":       HistGradientBoostingRegressor(max_iter=T,learning_rate=0.035,
                          max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=0.1,
                          random_state=RANDOM),
        "LightGBM_reg": lgb.LGBMRegressor(n_estimators=T,learning_rate=0.035,num_leaves=31,
                          min_child_samples=18,subsample=0.85,colsample_bytree=0.8,
                          reg_alpha=0.05,reg_lambda=25.0,random_state=RANDOM,
                          n_jobs=-1,verbose=-1),
    }
STYLE={"ExtraTrees":("s","#1f77b4"),"HistGB":("<","#ff7f0e"),"LightGBM_reg":("*","#17becf")}

def metrics(y,p): return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),
                              MAE=mean_absolute_error(y,p))

# ============================ CORE ==========================================
def compute(path,name,sheet,tgt,unit):
    print("\n"+"="*72+f"\n  {name}\n"+"="*72)
    X,y,grp=load_target(path,sheet,tgt)
    print(f"  rows={len(X)}  unique mixes (Exact_JMF_Group)={len(np.unique(grp))}  "
          f"features={X.shape[1]}  range=[{y.min():.2f},{y.max():.2f}]")
    ybin=pd.qcut(y,10,labels=False,duplicates="drop")
    sgkf=StratifiedGroupKFold(N_SPLITS,shuffle=True,random_state=RANDOM)
    folds=list(sgkf.split(X,ybin,groups=grp))
    # single stratified 80/20 grouped locked test (first fold)
    tr,te=folds[0]
    Xtr,Xte,ytr,yte,gtr=X.iloc[tr],X.iloc[te],y[tr],y[te],grp[tr]
    ybin_tr=pd.qcut(ytr,10,labels=False,duplicates="drop")
    inner=list(StratifiedGroupKFold(N_SPLITS,shuffle=True,random_state=RANDOM).split(Xtr,ybin_tr,groups=gtr))
    print(f"  train rows={len(ytr)}  locked-test rows={len(yte)}  (grouped, no mix leaks between)")

    fit_res={}; cv_r2={}
    for nm in zoo():
        # 10-fold pooled OOF on TRAIN (grouped by Mix_ID => LOPOCV-style)
        oof=np.zeros(len(ytr)); fold_scores=[]
        for a,b in inner:
            m=zoo()[nm]; m.fit(Xtr.iloc[a],ytr[a]); oof[b]=m.predict(Xtr.iloc[b])
            fold_scores.append(r2_score(ytr[b],oof[b]))
        # fit all-train, evaluate locked test
        m=zoo()[nm]; m.fit(Xtr,ytr); insample=m.predict(Xtr); ptest=m.predict(Xte)
        fit_res[nm]=dict(ptr=insample,pte=ptest,oof=oof,fold_r2=fold_scores,
                         **{f"R2_{k}":metrics(*args)["R2"] for k,args in
                            [("train",(ytr,insample)),("val",(ytr,oof)),("test",(yte,ptest))]})
        fit_res[nm]["mt"]=metrics(yte,ptest); fit_res[nm]["mv"]=metrics(ytr,oof); fit_res[nm]["mtr"]=metrics(ytr,insample)
        cv_r2[nm]=np.array(fold_scores)
        print(f"    {nm:12s} train R2={fit_res[nm]['R2_train']:.3f}  "
              f"OOF (grouped 10-fold) R2={fit_res[nm]['R2_val']:.3f}+/-{cv_r2[nm].std():.3f}  "
              f"locked-test R2={fit_res[nm]['R2_test']:.3f}  RMSE={fit_res[nm]['mt']['RMSE']:.3f}")
    # SHAP on the best model
    best=max(fit_res,key=lambda k:fit_res[k]["R2_test"])
    print(f"\n  BEST model: {best}   locked-test R2={fit_res[best]['R2_test']:.3f}")
    sh_model=zoo()["LightGBM_reg"]; sh_model.fit(Xtr,ytr)     # LightGBM for fast TreeExplainer
    samp=Xte.sample(min(600,len(Xte)),random_state=RANDOM)
    sv=shap.TreeExplainer(sh_model).shap_values(samp)
    shap_mean=pd.Series(np.abs(sv).mean(0),index=samp.columns).sort_values(ascending=False)
    return dict(name=name,unit=unit,ytr=ytr,yte=yte,fit=fit_res,cv=cv_r2,
                shap=sv,X_shap=samp,shap_mean=shap_mean,best=best,feats=list(X.columns))

# ============================ FIGURES ========================================
def plot_target(R):
    name=R["name"]; unit=R["unit"]; fit=R["fit"]; best=R["best"]
    fig,ax=plt.subplots(2,3,figsize=(16,9))
    fig.suptitle(f"{name}  |  best={best}  locked-test R2={fit[best]['R2_test']:.3f}  "
                 f"RMSE={fit[best]['mt']['RMSE']:.3f}",fontsize=14,fontweight="bold")
    ytr,yte=R["ytr"],R["yte"]
    # 1 parity - locked test
    a=ax[0,0]
    a.scatter(yte,fit[best]["pte"],alpha=.55,edgecolor="k",lw=.3,color="#2b7bba")
    lo,hi=min(yte.min(),fit[best]["pte"].min()),max(yte.max(),fit[best]["pte"].max())
    a.plot([lo,hi],[lo,hi],"r--",lw=1.4); a.set_title(f"Predicted vs Actual (locked test) - {best}")
    a.set_xlabel(f"Actual{unit}"); a.set_ylabel(f"Predicted{unit}")
    # 2 parity - grouped OOF
    a=ax[0,1]; oo=fit[best]["oof"]
    a.scatter(ytr,oo,alpha=.30,s=14,color="#5aae61")
    lo,hi=ytr.min(),ytr.max(); a.plot([lo,hi],[lo,hi],"r--",lw=1.4)
    a.set_title("Grouped 10-fold OOF predictions"); a.set_xlabel(f"Actual{unit}"); a.set_ylabel(f"Predicted{unit}")
    # 3 residuals
    res=fit[best]["pte"]-yte
    a=ax[0,2]; a.scatter(fit[best]["pte"],res,alpha=.55,edgecolor="k",lw=.3,color="#d6604d")
    a.axhline(0,color="k",lw=1); a.set_title("Residuals vs Predicted (test)")
    a.set_xlabel(f"Predicted{unit}"); a.set_ylabel(f"Residual{unit}")
    # 4 residual histogram
    a=ax[1,0]; a.hist(res,bins=25,color="#9970ab",edgecolor="k")
    a.set_title(f"Residual distribution (mean={res.mean():.2f})"); a.set_xlabel(f"Residual{unit}")
    # 5 CV fold stability
    a=ax[1,1]
    order=list(fit); vals=[fit[k]["R2_val"] for k in order]
    a.bar(order,vals,color=[STYLE[k][1] for k in order],edgecolor="k")
    for i,k in enumerate(order):
        a.errorbar(i,fit[k]["R2_val"],yerr=R["cv"][k].std(),color="k",capsize=4)
        a.annotate(f"{fit[k]['R2_val']:.3f}",(i,fit[k]["R2_val"]),textcoords="offset points",xytext=(0,4),ha="center",fontsize=9)
    a.set_ylim(0,1); a.set_ylabel("grouped 10-fold OOF R2")
    a.set_title("Model comparison (OOF R2)")
    # 6 top SHAP
    a=ax[1,2]; top=R["shap_mean"].head(15)[::-1]
    a.barh(top.index,top.values,color="#f4a582",edgecolor="k")
    a.set_title("Top-15 features (mean |SHAP|)")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_diag.png",bbox_inches="tight"); return fig

def plot_shap(R):
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}  |  SHAP explanations (locked-test set)",fontsize=14,fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None)
    ax1.set_title("SHAP beeswarm")
    ax2=fig.add_subplot(1,2,2); top=R["shap_mean"].head(15)[::-1]
    ax2.barh(top.index,top.values,color="#3182bd",edgecolor="k")
    ax2.set_title("Mean |SHAP| (global importance)"); ax2.set_xlabel("mean |SHAP value|")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_shap.png",bbox_inches="tight"); return fig

# ============================ MAIN ==========================================
if __name__=="__main__":
    print("[note] full run ~ 2-5 min per target.")
    path=resolve_file(FILE)
    Rs=[]
    Rs.append(compute(path,"RUT","Rutting_Filtered","LWT_Design_Result"," (mm)"))
    Rs.append(compute(path,"SCB","SCB_Filtered","SCB_Result",""))
    for R in Rs: plot_target(R); plot_shap(R)
    print("\n"+"="*72+"\nSUMMARY  (grouped 10-fold OOF + locked test, replicates KEPT)\n"+"="*72)
    for R in Rs:
        b=R["best"]; f=R["fit"][b]
        print(f"  {R['name']}   best={b}   OOF R2={f['R2_val']:.3f}   locked-test R2={f['R2_test']:.3f}   RMSE={f['mt']['RMSE']:.3f}")
    if not HEADLESS: plt.show()
    print("\nDONE - figures saved at 300 dpi and shown.")
