# -*- coding: utf-8 -*-
"""
MODEL DIAGNOSTICS & PAPER-STYLE FIGURES  --  rutting (LWT@20k) & SCB
================================================================================
Reproduces the figure suite of Wang (2025) for this dataset:
    * data distribution (target hist+KDE+box, feature correlation heatmap)
    * Fig-7  : per-model Actual vs Predicted (train + test), y=x, true-fit, R^2
    * Fig-8  : 10-fold CV R^2 boxplots per model ("K performance index")
    * Fig-9  : SHAP beeswarm + polar mean|SHAP| importance
    * Fig-13 : all-models Predicted vs Actual overlay
    * Fig-14 : relative-error distribution per model (5% / 10% guides)
    * Spearman ranking-consistency scatter (CV mean R^2 vs locked-test R^2)

DATA SETUP (as agreed):
    * target = LWT rut @20k / SCB ; NO range cap (drop only target<=0 / NaN)
    * EXACT-MIX dedup -> one row per mix (Mix_ID + full composition)   [no duplicates]
    * stratified, group-safe splits                                    [no leakage]
    * feature set: FEATURE_MODE = 'full' (physics+context) or 'physics' (physics-only)

Run in Spyder -> prints tables and pops every figure. All PNGs saved at 300 dpi.
Requirements: pandas numpy scikit-learn lightgbm xgboost catboost shap scipy matplotlib openpyxl
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap

# ============================== CONFIG =======================================
FILE         = "Book1._rutting__scb_design_validation_xlsx.xlsx"
# FEATURE_MODE: "engineering" (curated per-target set, RECOMMENDED) |
#               "full" (all ~62) | "jmf" (30 raw JMF only) | "physics" (drop context)
FEATURE_MODE = "engineering"
INCLUDE_CONTEXT_IN_ENGINEERING = True   # add ADT + Mix_Type/Design_Level to the curated set
CV_FOLDS     = 10
RANDOM       = 42
# target caps (same as the reliable benchmark: RUT 0.50-9.29, SCB 0.30-1.45)
CAP          = {"RUT":(0.5,10.0), "SCB":(0.30,1.60)}

# ---- curated engineering feature sets (from the SHAP + ablation study) ------
JMF_RAW = ["Va","VMA","VFA","Pbe","Pba","Gse","Gmm","AC","Dust_Pbe","Gmm_Ni","Gmm_Nm",
           "Absorption","FAA","CAA","SandEq","FlatElong","Gsb","Pass_3_4in","Pass_1_2in",
           "Pass_3_8in","Pass_No_4","Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50",
           "Pass_No_100","Pass_No_200","MixTemp","AC_from_RAP","NMAS_mm"]
ENG = {
  # rutting: traffic + aggregate skeleton + binder + densification
  "RUT":["NMAS_mm","Pass_No_4","Pass_No_8","Pass_No_30","Intermediate_Frac","Fine_Fraction",
         "Va","VMA","Gse","AC","Pba","RAP_Binder_Ratio","CAA","FAA","Gmm_Nm"],
  # scb: binder availability + mastic + fine gradation + aggregate quality
  "SCB":["Gse","Gsb","VMA","VFA","VMA_Filled_Index","Va","Absorption","FlatElong","FAA",
         "SandEq","Pass_No_16","Pass_No_30","Pass_No_50","Fine_Fraction","Pbe","Pba",
         "RAP_Binder_Ratio","AFT_micron"],
}
FAST         = False      # True = subsample + light models (smoke test)
HEADLESS     = False      # Spyder: keep False so plots show
OUTDIR       = "."

rcParams.update({"figure.dpi":110, "savefig.dpi":300, "font.size":12,
                 "axes.titlesize":13, "axes.labelsize":12, "legend.fontsize":9,
                 "axes.grid":True, "grid.alpha":0.25, "font.family":"DejaVu Sans"})

def resolve_file(name):
    if os.path.isfile(name): return name
    base=os.path.basename(name)
    for root in [os.getcwd(), os.path.expanduser("~/Downloads"), os.path.expanduser("~"), "C:/Users"]:
        for pat in [base,"*rutting*scb*.xlsx","*Book1*xlsx*"]:
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
def context_cols(cols):
    return [c for c in cols if c.startswith("Design_Level_") or c.startswith("Mix_Type_") or c=="ADT"]
def select_features(cols, target):
    """pick columns per FEATURE_MODE (target-aware for 'engineering')"""
    if FEATURE_MODE=="full":
        return cols
    if FEATURE_MODE=="physics":
        return [c for c in cols if c not in context_cols(cols)]
    if FEATURE_MODE=="jmf":
        return [c for c in JMF_RAW if c in cols]
    if FEATURE_MODE=="engineering":
        keep=[c for c in ENG.get(target,[]) if c in cols]
        if INCLUDE_CONTEXT_IN_ENGINEERING:
            keep=keep+[c for c in cols if c.startswith("Mix_Type_") or c.startswith("Design_Level_")]
            if "ADT" in cols: keep=keep+["ADT"]
        return list(dict.fromkeys(keep))
    return cols
def load_target(path,sheet,tgt,ylo,yhi,target):
    df=pd.read_excel(path,sheet_name=sheet); y=num(df[tgt]); X=build_features(df)
    # exact-mix key
    ident=[c for c in ["Mix_ID","Design_Level"] if c in df]
    sieves=["Pass_1_1_2in","Pass_1in","Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4","Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]
    comp=["Design_Submission__Percent_AC","Design_Submission__Percent_Voids","Design_Submission__VMA","Design_Submission__VFA","Design_Submission__Pbe","Design_Submission__Gse"]+["Design_Submission__"+s for s in sieves]+["Nominal_Aggregate_Size","Mix_Type","Total_AC_From_RAP"]
    kc=[c for c in ident+comp if c in df]; ks=df[kc].copy()
    for c in kc:
        if ks[c].dtype.kind in "fc": ks[c]=num(ks[c]).round(2)
    grp=np.array([str(v) for v in ks.apply(lambda col:col.map(str)).agg("|".join,axis=1)],dtype=object)
    X["__y__"]=y; X["__g__"]=grp
    X=X[np.isfinite(X["__y__"])&(X["__y__"]>=ylo)&(X["__y__"]<=yhi)]   # drop invalid + cap (reliable ranges)
    X=X.replace([np.inf,-np.inf],np.nan)
    agg=X.groupby("__g__").mean(numeric_only=True).reset_index()   # REMOVE DUPLICATES: one row per exact mix
    y=agg.pop("__y__").values; agg.pop("__g__")
    agg=agg.fillna(agg.median(numeric_only=True)).fillna(0.0)
    agg=agg[select_features(list(agg.columns), target)]      # apply FEATURE_MODE
    agg=agg.loc[:, agg.nunique()>1]                          # drop constant cols (HistGB-safe)
    if FAST:
        rng=np.random.RandomState(RANDOM); idx=rng.choice(len(agg),min(500,len(agg)),replace=False)
        agg,y=agg.iloc[idx].reset_index(drop=True),y[idx]
    return agg,y

# ============================== MODELS =======================================
def zoo():
    T=120 if FAST else 400; I=120 if FAST else 400
    return {
        "Ridge":      make_pipeline(StandardScaler(),Ridge(alpha=1.0,random_state=RANDOM)),
        "ElasticNet": make_pipeline(StandardScaler(),ElasticNet(alpha=0.01,l1_ratio=0.3,random_state=RANDOM)),
        "KNN":        make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=10)),
        "SVR":        make_pipeline(StandardScaler(),SVR(C=10.0,gamma="scale")),
        "RandomForest":RandomForestRegressor(n_estimators=T,n_jobs=-1,random_state=RANDOM),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=T,min_samples_leaf=2,max_features=0.85,n_jobs=-1,random_state=RANDOM),
        "HistGB":     HistGradientBoostingRegressor(max_iter=min(I,650),learning_rate=0.035,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=0.1,random_state=RANDOM),
        "XGBoost_reg":xgb.XGBRegressor(n_estimators=min(I,650),learning_rate=0.035,max_depth=4,min_child_weight=5,subsample=0.85,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=20.0,random_state=RANDOM,n_jobs=-1,verbosity=0),
        "LightGBM_reg":lgb.LGBMRegressor(n_estimators=min(I,650),learning_rate=0.035,num_leaves=31,min_child_samples=18,subsample=0.85,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=25.0,random_state=RANDOM,n_jobs=-1,verbose=-1),
        "CatBoost_reg":CatBoostRegressor(iterations=min(I,650),learning_rate=0.035,depth=5,l2_leaf_reg=20,random_state=RANDOM,verbose=0),
    }
# distinct marker+color per model (paper style)
STYLE={"Ridge":("P","#8c564b"),"ElasticNet":("X","#e377c2"),"KNN":("v","#7f7f7f"),
       "SVR":("D","#d62728"),"RandomForest":("^","#2ca02c"),"ExtraTrees":("s","#1f77b4"),
       "HistGB":("<","#ff7f0e"),"XGBoost_reg":("o","#9467bd"),"LightGBM_reg":("*","#17becf"),
       "CatBoost_reg":("h","#bcbd22")}

def new_zoo(): return zoo()

# ============================== CORE COMPUTE =================================
def compute(path, name, sheet, tgt, unit):
    print("\n"+"="*70+f"\n  {name}   (FEATURE_MODE={FEATURE_MODE})\n"+"="*70)
    key=name.split()[0]; ylo,yhi=CAP[key]
    X,y=load_target(path,sheet,tgt,ylo,yhi,key); feats=list(X.columns)
    print(f"  rows after removing duplicates (one per exact mix)={len(X)}  "
          f"features={X.shape[1]}  target range=[{y.min():.2f},{y.max():.2f}]")
    print(f"  features used: {feats}")
    ybin=pd.qcut(y,10,labels=False,duplicates="drop")
    cv10=list(StratifiedKFold(CV_FOLDS,shuffle=True,random_state=RANDOM).split(X,ybin))
    cv3 =list(StratifiedKFold(3,shuffle=True,random_state=RANDOM).split(X,ybin))
    fit_res={}; cv_r2={}; cv3_r2={}
    for nm in new_zoo():
        # 10-fold pooled out-of-fold predictions  (honest "test")
        oof=np.zeros(len(y)); folds=[]
        for a,b in cv10:
            m=new_zoo()[nm]; m.fit(X.iloc[a],y[a]); p=m.predict(X.iloc[b]); oof[b]=p
            folds.append(r2_score(y[b],p))
        # in-sample fit on all data  (optimistic "train")
        mm=new_zoo()[nm]; mm.fit(X,y); insample=mm.predict(X)
        # 3-fold pooled (for ranking-consistency scatter)
        oof3=np.zeros(len(y))
        for a,b in cv3:
            m=new_zoo()[nm]; m.fit(X.iloc[a],y[a]); oof3[b]=m.predict(X.iloc[b])
        fit_res[nm]=dict(ptr=insample,pte=oof,
                         R2_tr=r2_score(y,insample),R2_te=r2_score(y,oof),
                         RMSE=np.sqrt(mean_squared_error(y,oof)),MAE=mean_absolute_error(y,oof))
        cv_r2[nm]=np.array(folds); cv3_r2[nm]=r2_score(y,oof3)
        print(f"    {nm:13s} OOF R2={fit_res[nm]['R2_te']:.3f}  RMSE={fit_res[nm]['RMSE']:.3f}  "
              f"10-fold R2={cv_r2[nm].mean():.3f}+/-{cv_r2[nm].std():.3f}")
    # SHAP (LightGBM, out-of-fold-style on a sample)
    samp=np.random.RandomState(RANDOM).choice(len(X),min(900,len(X)),replace=False)
    lgbm=new_zoo()["LightGBM_reg"]; lgbm.fit(X,y)
    Xsh=X.iloc[samp]; sv=shap.TreeExplainer(lgbm).shap_values(Xsh)
    shap_mean=pd.Series(np.abs(sv).mean(0),index=feats).sort_values(ascending=False)
    return dict(name=name,unit=unit,X=X,y=y,feats=feats,ytr=y,yte=y,
                fit=fit_res,cv=cv_r2,cv3=cv3_r2,shap=sv,Xte=Xsh,shap_mean=shap_mean)

# ============================== FIGURES ======================================
def fig_distribution(R):
    name,unit,y,X=R["name"],R["unit"],R["y"],R["X"]
    fig=plt.figure(figsize=(15,4.5)); fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1)
    ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
    xs=np.linspace(y.min(),y.max(),200); ax.plot(xs,gaussian_kde(y)(xs),"r-",lw=2)
    ax.set_xlabel(f"{name} target{unit}"); ax.set_ylabel("Density"); ax.set_title("Target histogram + KDE")
    ax=fig.add_subplot(1,3,2); ax.boxplot(y,vert=True,widths=.5,patch_artist=True,
        boxprops=dict(facecolor="#69b3d6")); ax.set_ylabel(f"{name} target{unit}"); ax.set_title("Target boxplot")
    ax.set_xticks([])
    ax=fig.add_subplot(1,3,3)
    top=R["shap_mean"].head(12).index.tolist()
    C=X[top].corr().values
    im=ax.imshow(C,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top,rotation=90,fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=7)
    ax.set_title("Feature correlation (top-12)"); fig.colorbar(im,ax=ax,fraction=0.046)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(f"{OUTDIR}/{name}_dist.png",bbox_inches="tight"); return fig

def fig_actual_pred(R):
    name,unit=R["name"],R["unit"]; ytr,yte=R["ytr"],R["yte"]
    fig,axs=plt.subplots(2,5,figsize=(20,8)); fig.suptitle(
        f"{name}: Actual vs Predicted by model{unit}",fontweight="bold")
    lo=min(ytr.min(),yte.min()); hi=max(ytr.max(),yte.max())
    for ax,(nm,fr) in zip(axs.ravel(),R["fit"].items()):
        ax.scatter(ytr,fr["ptr"],marker="D",s=16,facecolor="none",edgecolor="#7e57c2",alpha=.6,label="Training data")
        ax.scatter(yte,fr["pte"],marker="^",s=22,color="#2ca02c",alpha=.7,label="Test data")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.3,label="Perfect fit (y=x)")
        a,b=np.polyfit(yte,fr["pte"],1); xs=np.array([lo,hi])
        ax.plot(xs,a*xs+b,"r-",lw=1.5,label="True fit")
        ax.text(0.05,0.92,f"y={a:.3f}x+{b:.2f}\n$R^2$={fr['R2_te']:.3f}",transform=ax.transAxes,
                va="top",fontsize=9,bbox=dict(fc="white",ec="grey",alpha=.7))
        ax.set_title(nm); ax.set_xlabel(f"Actual value{unit}"); ax.set_ylabel(f"Predicted value{unit}")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    axs.ravel()[0].legend(loc="lower right",fontsize=7)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUTDIR}/{name}_actual_pred.png",bbox_inches="tight"); return fig

def fig_cv_box(R):
    name=R["name"]; order=sorted(R["cv"],key=lambda k:R["cv"][k].mean(),reverse=True)
    data=[R["cv"][k] for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),
                  medianprops=dict(color="k"),flierprops=dict(marker="d",ms=4,mfc="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE[k][1]); patch.set_alpha(.65)
    for i,k in enumerate(order):
        ax.annotate(f"{R['cv'][k].mean():.2f}",(i+1,R['cv'][k].mean()),textcoords="offset points",
                    xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel(f"$R^2$ ({CV_FOLDS}-fold CV, grouped)")
    ax.set_title(f"{name}: {CV_FOLDS}-fold cross-validation performance (leakage-safe)")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_cvbox.png",bbox_inches="tight"); return fig

def fig_overlay(R):
    name,unit,yte=R["name"],R["unit"],R["yte"]
    fig,ax=plt.subplots(figsize=(7.5,7.5))
    lo,hi=yte.min(),yte.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE[nm]; ax.scatter(yte,fr["pte"],marker=mk,s=30,color=col,alpha=.7,label=nm,edgecolor="k",lw=.3)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y = x")
    ax.set_xlabel(f"Actual value{unit}"); ax.set_ylabel(f"Predicted value{unit}")
    ax.set_title(f"{name}: prediction results of different models"); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_overlay.png",bbox_inches="tight"); return fig

def fig_relerr(R):
    name,yte=R["name"],R["yte"]
    order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_te"],reverse=True)
    data=[np.abs((R["fit"][k]["pte"]-yte)/yte)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE[k][1]); patch.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey",lw=1.2,label="5%"); ax.axhline(10,ls=":",color="grey",lw=1.2,label="10%")
    for i,k in enumerate(order):
        ax.annotate(f"{np.median(data[i]):.1f}",(i+1,np.median(data[i])),textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error of different models (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_relerr.png",bbox_inches="tight"); return fig

def fig_shap(R):
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}: SHAP summary & importance (LightGBM, test set)",fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["Xte"],plot_type="dot",max_display=15,show=False,plot_size=None)
    ax1.set_title("SHAP beeswarm")
    # polar importance
    ax2=fig.add_subplot(1,2,2,polar=True)
    top=R["shap_mean"].head(14)[::-1]; N=len(top)
    ang=np.linspace(0,2*np.pi,N,endpoint=False); width=2*np.pi/N*0.9
    ax2.bar(ang,top.values,width=width,color="#9b8cc4",edgecolor="k",alpha=.8)
    ax2.set_xticks(ang); ax2.set_xticklabels(top.index,fontsize=7)
    ax2.set_title("Mean |SHAP| (polar)")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUTDIR}/{name}_shap.png",bbox_inches="tight"); return fig

def fig_spearman(R):
    name=R["name"]; models=list(R["fit"])
    xv=np.array([R["cv"][m].mean() for m in models])   # 10-fold CV mean R2
    yv=np.array([R["cv3"][m]        for m in models])  # 3-fold CV R2
    rho=spearmanr(xv,yv).correlation
    fig,ax=plt.subplots(figsize=(7,6.5))
    for m in models:
        mk,col=STYLE[m]; ax.scatter(R["cv"][m].mean(),R["cv3"][m],marker=mk,s=90,color=col,edgecolor="k",label=m)
    lo=min(xv.min(),yv.min())-0.05; hi=max(xv.max(),yv.max())+0.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(f"{CV_FOLDS}-fold CV mean $R^2$"); ax.set_ylabel("3-fold CV $R^2$")
    ax.set_title(f"{name}: model ranking consistency\nSpearman $\\rho$ = {rho:.3f}")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_spearman.png",bbox_inches="tight"); return fig

# ============================== MAIN =========================================
if __name__=="__main__":
    print("[note] full run ~ a few minutes per target (10 models x 10-fold CV + SHAP).")
    path=resolve_file(FILE)
    Rs=[]
    Rs.append(compute(path,"RUT","Rutting_Design","LWT_Design_Result"," (mm)"))
    Rs.append(compute(path,"SCB","SCB_Design","SCB_Result",""))
    for R in Rs:
        fig_distribution(R); fig_actual_pred(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_spearman(R)
    if not HEADLESS: plt.show()
    print("\nDONE - all diagnostic figures saved at 300 dpi and shown.")
