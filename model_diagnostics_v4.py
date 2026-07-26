# -*- coding: utf-8 -*-
"""
MODEL DIAGNOSTICS v4  --  RUT + SCB  (LOPOCV-style grouped 10-fold on KEEP_REPLICATES file)
================================================================================
Data source
-----------
Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx
    Pre-applied to Book1: RUT 0.5-10, SCB 0.30-1.25, OGFC/SMA/Thin Lift removed.
    REPLICATES KEPT (RUT 4591 rows / 1640 mixes ; SCB 3052 rows / 1271 mixes).
    Column Exact_JMF_Group  =  Mix_ID + full composition  -> proper LOPOCV grouping.

Method
------
    * 10 supervised regressors benchmarked
    * Stratified GroupKFold, 10 folds  ->  pooled OOF R^2  (LOPOCV-style, no leakage)
    * Feature sets by target: RUT vs SCB curated engineering set + PG_High from Custom_Name
    * All 7 paper-style figures produced per target:
         (1) data distribution (hist+KDE+box+correlation heatmap)
         (2) actual-vs-predicted grid for every model (train/test, y=x, true-fit, R^2)
         (3) 10-fold CV R^2 boxplots per model
         (4) all-model overlay of test predictions
         (5) relative-error % boxplots per model (5%/10% guides)
         (6) SHAP beeswarm + polar mean|SHAP|
         (7) Spearman ranking-consistency scatter (CV mean R^2 vs test R^2)

Run in Spyder -> prints per-model R^2 table and pops every figure.
Requirements: pandas numpy scikit-learn lightgbm xgboost catboost shap scipy matplotlib openpyxl
"""
import os, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap

# =============================== CONFIG =====================================
FILE      = "Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx"
CV_FOLDS  = 10
RANDOM    = 42
FAST      = False        # True = smaller trees, fewer rows for smoke test
HEADLESS  = False        # Spyder: keep False so plots pop up
OUTDIR    = "."

rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":12,
                 "axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":9,
                 "axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})

def resolve_file(name):
    if os.path.isfile(name): return name
    base=os.path.basename(name)
    for root in [os.getcwd(),os.path.expanduser("~/Downloads"),
                 os.path.expanduser("~"),"C:/Users"]:
        for pat in [base,"*KEEP_REPLICATES*.xlsx","*filtered*RUT*.xlsx"]:
            hit=glob.glob(os.path.join(root,"**",pat),recursive=True)
            if hit: print(f"[info] using workbook: {hit[0]}"); return hit[0]
    raise FileNotFoundError(f"Set FILE at top of script to the full path of '{base}'.")

# =========================== FEATURES ========================================
NMAS_MAP={"1/2 in.":12.5,"3/4 in.":19.0,"1 in.":25.0,"3/8 in.":9.5,"1/4 in.":6.25}
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
    X["SandEq"]=pick(df,A+"Sand_Equivalent");X["FlatElong"]=pick(df,A+"Flat_Elongated")
    X["Gsb"]=pick(df,A+"Bulk_Gravity")
    for s in ["Pass_3_4in","Pass_1_2in","Pass_3_8in","Pass_No_4","Pass_No_8",
              "Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]:
        X[s]=pick(df,P+s)
    X["ADT"]=pick(df,"ADT");X["MixTemp"]=pick(df,"Mix_Temperature")
    X["AC_from_RAP"]=pick(df,"Total_AC_From_RAP");X["Production_Rate"]=pick(df,"Production_Rate")
    if "Nominal_Aggregate_Size" in df.columns:
        v=df["Nominal_Aggregate_Size"]
        X["NMAS_mm"]=v.map(NMAS_MAP) if v.dtype==object else num(v)
    # PG grade parsed from Custom_Name
    if "Custom_Name" in df.columns:
        pg=df["Custom_Name"].apply(lambda s: re.search(r'(\d{2,3})\s*[-\u2013]\s*(\d{2})',str(s).upper()))
        X["PG_High"]=[float(m.group(1)) if m else np.nan for m in pg]
        X["Polymer"]=df["Custom_Name"].apply(lambda s: 1.0 if re.search(r'SBS|POLY|LATEX|MODIF',str(s).upper()) else 0.0)
        X["PG_known"]=X["PG_High"].notna().astype(float); X["PG_High"]=X["PG_High"].fillna(0.0)
    for c in ["Design_Level","Mix_Type"]:
        if c in df.columns:
            d=pd.get_dummies(df[c].astype(str),prefix=c).astype(float)
            X=pd.concat([X,d.set_index(X.index)],axis=1)
    # engineered proxies (kept for interpretation; RBR/AFT/etc.)
    X["RAP_Binder_Ratio"]=X["AC_from_RAP"]/X["AC"].replace(0,np.nan)
    X["Fine_Fraction"]=X["Pass_No_8"]-X["Pass_No_200"]
    X["Intermediate_Frac"]=X["Pass_No_4"]-X["Pass_No_8"]
    X["VMA_Filled_Index"]=X["VMA"]*X["VFA"]/100
    sa=sum(X[k]*v for k,v in SA_FACTOR.items())/100+0.41
    X["AFT_micron"]=(X["Pbe"]/100)/(sa*X["Gsb"].replace(0,np.nan))*1000
    return X

# per-target curated engineering feature set
ENG = {
  "RUT":["NMAS_mm","Pass_No_4","Pass_No_8","Pass_No_30","Intermediate_Frac","Fine_Fraction",
         "Va","VMA","Gse","AC","Pba","RAP_Binder_Ratio","CAA","FAA","Gmm_Nm",
         "PG_High","Polymer","PG_known","Production_Rate"],
  "SCB":["Gse","Gsb","VMA","VFA","VMA_Filled_Index","Va","Absorption","FlatElong","FAA",
         "SandEq","Pass_No_16","Pass_No_30","Pass_No_50","Fine_Fraction","Pbe","Pba",
         "RAP_Binder_Ratio","AFT_micron","PG_High","PG_known","Polymer"],
}
def context_cols(cols):
    return [c for c in cols if c.startswith("Design_Level_") or c.startswith("Mix_Type_") or c=="ADT"]
def select_features(cols,target):
    keep=[c for c in ENG.get(target,[]) if c in cols]
    keep+=context_cols(cols)
    return list(dict.fromkeys(keep))

def load_target(path,sheet,tgt,target_key):
    df=pd.read_excel(path,sheet_name=sheet)
    y=num(df[tgt]).values
    grp=df["Exact_JMF_Group"].astype(str).values   # pre-built grouping (LOPOCV-safe)
    X=build_features(df)
    X=X[select_features(list(X.columns),target_key)]
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    if FAST:
        rng=np.random.RandomState(RANDOM); pick=rng.choice(len(X),min(1000,len(X)),replace=False)
        X,y,grp=X.iloc[pick].reset_index(drop=True),y[pick],grp[pick]
    return X,y,grp

# ============================ MODELS =========================================
def zoo():
    T=200 if FAST else 400; I=200 if FAST else 400
    return {
        "Ridge":       make_pipeline(StandardScaler(),Ridge(alpha=1.0,random_state=RANDOM)),
        "ElasticNet":  make_pipeline(StandardScaler(),ElasticNet(alpha=0.01,l1_ratio=0.3,random_state=RANDOM)),
        "KNN":         make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=10)),
        "SVR":         make_pipeline(StandardScaler(),SVR(C=10.0,gamma="scale")),
        "RandomForest":RandomForestRegressor(n_estimators=T,min_samples_leaf=2,max_features=0.85,n_jobs=-1,random_state=RANDOM),
        "ExtraTrees":  ExtraTreesRegressor(n_estimators=T,min_samples_leaf=2,max_features=0.85,n_jobs=-1,random_state=RANDOM),
        "GradBoost":   GradientBoostingRegressor(n_estimators=min(I,300),learning_rate=0.05,max_depth=3,random_state=RANDOM),
        "XGBoost":     xgb.XGBRegressor(n_estimators=I,learning_rate=0.04,max_depth=5,subsample=0.85,
                                        colsample_bytree=0.8,random_state=RANDOM,n_jobs=-1,verbosity=0),
        "LightGBM":    lgb.LGBMRegressor(n_estimators=I,learning_rate=0.04,num_leaves=31,subsample=0.85,
                                        colsample_bytree=0.8,random_state=RANDOM,n_jobs=-1,verbose=-1),
        "CatBoost":    CatBoostRegressor(iterations=I,learning_rate=0.04,depth=6,random_state=RANDOM,verbose=0),
    }
STYLE={"Ridge":("P","#8c564b"),"ElasticNet":("X","#e377c2"),"KNN":("v","#7f7f7f"),
       "SVR":("D","#d62728"),"RandomForest":("^","#2ca02c"),"ExtraTrees":("s","#1f77b4"),
       "GradBoost":("<","#ff7f0e"),"XGBoost":("o","#9467bd"),"LightGBM":("*","#17becf"),
       "CatBoost":("h","#bcbd22")}

def metrics(y,p):
    return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p))

# ============================ CORE ===========================================
def compute(path,name,sheet,tgt,unit):
    print("\n"+"="*70+f"\n  {name}   (LOPOCV-style grouped {CV_FOLDS}-fold, replicates KEPT)\n"+"="*70)
    X,y,grp=load_target(path,sheet,tgt,name)
    feats=list(X.columns)
    print(f"  rows={len(X)}  unique mixes (Exact_JMF_Group)={len(np.unique(grp))}  "
          f"features={X.shape[1]}  target range=[{y.min():.2f},{y.max():.2f}]")
    print(f"  features used: {feats}")
    ybin=pd.qcut(y,10,labels=False,duplicates="drop")
    sgkf=StratifiedGroupKFold(CV_FOLDS,shuffle=True,random_state=RANDOM)
    folds=list(sgkf.split(X,ybin,groups=grp))
    # use fold-0 as the locked test slice for parity figures
    tr0,te0=folds[0]
    fit_res={}; cv_r2={}
    for nm in zoo():
        # 10-fold pooled OOF on FULL data (LOPOCV-style: replicates never leak)
        oof=np.zeros(len(y)); fold_scores=[]
        for a,b in folds:
            m=zoo()[nm]; m.fit(X.iloc[a],y[a]); oof[b]=m.predict(X.iloc[b])
            fold_scores.append(r2_score(y[b],oof[b]))
        # fit train-part, predict test slice for parity plots
        m=zoo()[nm]; m.fit(X.iloc[tr0],y[tr0])
        ptr=m.predict(X.iloc[tr0]); pte=m.predict(X.iloc[te0])
        cv_r2[nm]=np.array(fold_scores)
        oof_metrics=metrics(y,oof); test_metrics=metrics(y[te0],pte); train_metrics=metrics(y[tr0],ptr)
        fit_res[nm]=dict(oof=oof,ptr=ptr,pte=pte,fold_r2=fold_scores,
                         R2_oof=oof_metrics["R2"],RMSE_oof=oof_metrics["RMSE"],MAE_oof=oof_metrics["MAE"],
                         R2_te=test_metrics["R2"],RMSE_te=test_metrics["RMSE"],MAE_te=test_metrics["MAE"],
                         R2_tr=train_metrics["R2"])
        print(f"    {nm:13s} OOF R2={oof_metrics['R2']:.3f}  RMSE={oof_metrics['RMSE']:.3f}  "
              f"{CV_FOLDS}-fold R2={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}")
    # SHAP on the winning tree model (LightGBM for speed)
    best_tree="LightGBM"
    lgbm=zoo()[best_tree]; lgbm.fit(X.iloc[tr0],y[tr0])
    samp=X.iloc[te0].sample(min(600,len(te0)),random_state=RANDOM)
    sv=shap.TreeExplainer(lgbm).shap_values(samp)
    shap_mean=pd.Series(np.abs(sv).mean(0),index=samp.columns).sort_values(ascending=False)
    return dict(name=name,unit=unit,X=X,y=y,tr=tr0,te=te0,feats=feats,
                fit=fit_res,cv=cv_r2,shap=sv,X_shap=samp,shap_mean=shap_mean)

# ============================ FIGURES ========================================
def fig_distribution(R):
    name,unit,y,X=R["name"],R["unit"],R["y"],R["X"]
    fig=plt.figure(figsize=(15,4.5))
    fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1)
    ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
    xs=np.linspace(y.min(),y.max(),200); ax.plot(xs,gaussian_kde(y)(xs),"r-",lw=2)
    ax.set_xlabel(f"{name} target{unit}"); ax.set_ylabel("Density"); ax.set_title("Target histogram + KDE")
    ax=fig.add_subplot(1,3,2); ax.boxplot(y,vert=True,widths=.5,patch_artist=True,
        boxprops=dict(facecolor="#69b3d6")); ax.set_ylabel(f"{name} target{unit}"); ax.set_title("Target boxplot"); ax.set_xticks([])
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
    name,unit=R["name"],R["unit"]; ytr,yte=R["y"][R["tr"]],R["y"][R["te"]]
    fig,axs=plt.subplots(2,5,figsize=(20,8))
    fig.suptitle(f"{name}: Actual vs Predicted by model{unit}",fontweight="bold")
    lo=min(ytr.min(),yte.min()); hi=max(ytr.max(),yte.max())
    for ax,(nm,fr) in zip(axs.ravel(),R["fit"].items()):
        ax.scatter(ytr,fr["ptr"],marker="D",s=16,facecolor="none",edgecolor="#7e57c2",alpha=.6,label="Training data")
        ax.scatter(yte,fr["pte"],marker="^",s=22,color="#2ca02c",alpha=.7,label="Test data")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.3,label="Perfect fit (y=x)")
        if len(yte)>=2 and np.std(yte)>0:
            a,b=np.polyfit(yte,fr["pte"],1); xs=np.array([lo,hi])
            ax.plot(xs,a*xs+b,"r-",lw=1.5,label="True fit")
            ax.text(0.05,0.92,f"y={a:.3f}x+{b:.2f}\n$R^2$={fr['R2_te']:.3f}",transform=ax.transAxes,
                    va="top",fontsize=9,bbox=dict(fc="white",ec="grey",alpha=.7))
        ax.set_title(nm); ax.set_xlabel(f"Actual value{unit}"); ax.set_ylabel(f"Predicted value{unit}")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    axs.ravel()[0].legend(loc="lower right",fontsize=7)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_actual_pred.png",bbox_inches="tight"); return fig

def fig_cv_box(R):
    name=R["name"]; order=sorted(R["cv"],key=lambda k:R["cv"][k].mean(),reverse=True)
    data=[R["cv"][k] for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),
                  medianprops=dict(color="k"),flierprops=dict(marker="d",ms=4,mfc="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE[k][1]); patch.set_alpha(.65)
    for i,k in enumerate(order):
        ax.annotate(f"{R['cv'][k].mean():.2f}",(i+1,R['cv'][k].mean()),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right")
    ax.set_ylabel(f"$R^2$ ({CV_FOLDS}-fold grouped OOF, replicates kept)")
    ax.set_title(f"{name}: {CV_FOLDS}-fold cross-validation performance (LOPOCV-style)")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_cvbox.png",bbox_inches="tight"); return fig

def fig_overlay(R):
    name,unit,yte=R["name"],R["unit"],R["y"][R["te"]]
    fig,ax=plt.subplots(figsize=(7.5,7.5))
    lo,hi=yte.min(),yte.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE[nm]
        ax.scatter(yte,fr["pte"],marker=mk,s=30,color=col,alpha=.7,label=nm,edgecolor="k",lw=.3)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y = x")
    ax.set_xlabel(f"Actual value{unit}"); ax.set_ylabel(f"Predicted value{unit}")
    ax.set_title(f"{name}: prediction results of different models"); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_overlay.png",bbox_inches="tight"); return fig

def fig_relerr(R):
    name,yte=R["name"],R["y"][R["te"]]
    order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_te"],reverse=True)
    yte_safe=np.where(np.abs(yte)<1e-6,1e-6,yte)
    data=[np.abs((R["fit"][k]["pte"]-yte)/yte_safe)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE[k][1]); patch.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey",lw=1.2,label="5%"); ax.axhline(10,ls=":",color="grey",lw=1.2,label="10%")
    for i,k in enumerate(order):
        ax.annotate(f"{np.median(data[i]):.1f}",(i+1,np.median(data[i])),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error of different models (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_relerr.png",bbox_inches="tight"); return fig

def fig_shap(R):
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}: SHAP summary & importance (LightGBM, test set)",fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None)
    ax1.set_title("SHAP beeswarm")
    ax2=fig.add_subplot(1,2,2,polar=True)
    top=R["shap_mean"].head(14)[::-1]; N=len(top)
    ang=np.linspace(0,2*np.pi,N,endpoint=False); width=2*np.pi/N*0.9
    ax2.bar(ang,top.values,width=width,color="#9b8cc4",edgecolor="k",alpha=.8)
    ax2.set_xticks(ang); ax2.set_xticklabels(top.index,fontsize=7)
    ax2.set_title("Mean |SHAP| (polar)")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_shap.png",bbox_inches="tight"); return fig

def fig_spearman(R):
    name=R["name"]; models=list(R["fit"])
    xv=np.array([R["cv"][m].mean() for m in models])
    yv=np.array([R["fit"][m]["R2_te"] for m in models])
    rho=spearmanr(xv,yv).correlation
    fig,ax=plt.subplots(figsize=(7,6.5))
    for m in models:
        mk,col=STYLE[m]
        ax.scatter(R["cv"][m].mean(),R["fit"][m]["R2_te"],marker=mk,s=90,color=col,edgecolor="k",label=m)
    lo=min(xv.min(),yv.min())-0.05; hi=max(xv.max(),yv.max())+0.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(f"{CV_FOLDS}-fold CV mean $R^2$"); ax.set_ylabel("Locked-test $R^2$")
    ax.set_title(f"{name}: model ranking consistency\nSpearman $\\rho$ = {rho:.3f}")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_spearman.png",bbox_inches="tight"); return fig

# ============================ MAIN ===========================================
if __name__=="__main__":
    print("[note] full run ~ 5-10 min per target (10 models x 10-fold grouped CV + SHAP).")
    path=resolve_file(FILE)
    Rs=[]
    Rs.append(compute(path,"RUT","Rutting_Filtered","LWT_Design_Result"," (mm)"))
    Rs.append(compute(path,"SCB","SCB_Filtered","SCB_Result",""))
    for R in Rs:
        fig_distribution(R); fig_actual_pred(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_spearman(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*70+"\nSUMMARY (grouped 10-fold OOF R2, replicates KEPT, LOPOCV-style)\n"+"="*70)
    for R in Rs:
        best=max(R["fit"],key=lambda k:R["fit"][k]["R2_oof"])
        f=R["fit"][best]
        print(f"  {R['name']}   best={best}   OOF R2={f['R2_oof']:.3f}   RMSE={f['RMSE_oof']:.3f}")
    print("\nDONE - all 7 diagnostic figures per target saved at 300 dpi and shown.")
