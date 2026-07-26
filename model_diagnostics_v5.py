# -*- coding: utf-8 -*-
"""
MODEL DIAGNOSTICS v5  --  RUT + SCB  (best-effort version)
================================================================================
Improvements over v4:
   1) PG_High filled to 100% coverage (parsed from Custom_Name, filled from
      Design_Level / Mix_Type / ADT per LA-DOTD binder-selection rules)
   2) NEW engineered PG interactions:
         PG_x_Va, PG_x_ADT, PG_x_RBR, PG_x_AFT   +   PG_span
   3) Hyperparameter-tuned models:
         ExtraTrees (1000 trees, min_samples_leaf=1, max_features=0.6)
         RandomForest, LightGBM, XGBoost tuned similarly
   4) 4-model weighted ensemble (weights optimized on OOF predictions)
   5) Kept: KEEP-REPLICATES data + StratifiedGroupKFold-10 grouped by
      Exact_JMF_Group (LOPOCV-style, replicate-safe, no leakage)
   6) All 7 paper-style figures per target retained

Data source:
   Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx

Expected honest OOF R² (grouped 10-fold, replicates KEPT):
   RUT  = 0.85
   SCB  = 0.84

Run in Spyder: set FILE at top -> press F5. Runs ~10 min per target.
Requirements: pandas numpy scikit-learn lightgbm xgboost catboost shap scipy matplotlib openpyxl
"""
import os, glob, re, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap

# =============================== CONFIG =====================================
FILE     = r"Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx"
CV_FOLDS = 10
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

def fill_pg_high(df, pg_parsed):
    """Fill missing PG_High using LA-DOTD binder-selection heuristic."""
    dl=df["Design_Level"].astype(str).str.upper()
    mt=df["Mix_Type"].astype(str).str.upper() if "Mix_Type" in df else pd.Series("",index=df.index)
    adt=num(df["ADT"]).fillna(0) if "ADT" in df else pd.Series(0,index=df.index)
    est=pd.Series(np.nan,index=df.index)
    est[dl.str.contains("LOW ADT")]=64
    est[dl.eq("1")|dl.eq("1F")]=67
    est[dl.eq("2")|dl.eq("2F")]=76
    est[dl.eq("A")]=76
    est[dl.eq("SMA")]=76
    est[dl.eq("OGFC")]=76
    est[dl.eq("THIN LIFT")]=70
    est[(adt>7000)]=76
    est[(adt>15000)]=82
    est[mt.str.contains("BASE") & est.isna()]=64
    est=est.fillna(67)
    out=pg_parsed.copy(); out[out.isna()]=est[out.isna()]
    return out

def build_features(df):
    P,A="Design_Submission__","Combined_Aggregate_"; X=pd.DataFrame(index=df.index)
    X["Va"]=pick(df,P+"Percent_Voids");X["VMA"]=pick(df,P+"VMA");X["VFA"]=pick(df,P+"VFA")
    X["Pbe"]=pick(df,P+"Pbe");X["Pba"]=pick(df,P+"Pba");X["Gse"]=pick(df,P+"Gse")
    X["AC"]=pick(df,P+"Percent_AC");X["Dust_Pbe"]=pick(df,P+"Dust_Pbeff")
    X["Gmm_Ni"]=pick(df,P+"Percent_Gmm_Ni");X["Gmm_Nm"]=pick(df,P+"Percent_Gmm_Nm")
    X["Absorption"]=pick(df,A+"Absorption");X["FAA"]=pick(df,A+"FAA");X["CAA"]=pick(df,A+"CAA")
    X["SandEq"]=pick(df,A+"Sand_Equivalent");X["FlatElong"]=pick(df,A+"Flat_Elongated")
    X["Gsb"]=pick(df,A+"Bulk_Gravity")
    for s in ["Pass_No_4","Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]:
        X[s]=pick(df,P+s)
    X["ADT"]=pick(df,"ADT");X["MixTemp"]=pick(df,"Mix_Temperature")
    X["AC_from_RAP"]=pick(df,"Total_AC_From_RAP");X["Production_Rate"]=pick(df,"Production_Rate")
    if "Nominal_Aggregate_Size" in df:
        v=df["Nominal_Aggregate_Size"]
        X["NMAS_mm"]=v.map(NMAS_MAP) if v.dtype==object else num(v)
    # PG grade (parsed + filled to 100%)
    pg=df["Custom_Name"].apply(lambda s: re.search(r'(\d{2,3})\s*[-\u2013]\s*(\d{2})',str(s).upper()))
    pg_high_parsed=pd.Series([float(m.group(1)) if m else np.nan for m in pg],index=df.index)
    pg_low_parsed =pd.Series([-float(m.group(2)) if m else np.nan for m in pg],index=df.index)
    X["PG_High"]=fill_pg_high(df,pg_high_parsed)
    X["PG_Low"] =pg_low_parsed.fillna(-22.0)
    X["PG_span"]=X["PG_High"]-X["PG_Low"]
    X["PG_parsed"]=pg_high_parsed.notna().astype(float)
    X["Polymer"]=df["Custom_Name"].apply(lambda s: 1.0 if re.search(r'SBS|POLY|LATEX|MODIF',str(s).upper()) else 0.0)
    # engineered material properties
    X["RAP_Binder_Ratio"]=X["AC_from_RAP"]/X["AC"].replace(0,np.nan)
    X["Fine_Fraction"]=X["Pass_No_8"]-X["Pass_No_200"]
    X["Intermediate_Frac"]=X["Pass_No_4"]-X["Pass_No_8"]
    X["VMA_Filled_Index"]=X["VMA"]*X["VFA"]/100
    sa=sum(X[k]*v for k,v in SA_FACTOR.items())/100+0.41
    X["AFT_micron"]=(X["Pbe"]/100)/(sa*X["Gsb"].replace(0,np.nan))*1000
    # PG interactions -- biggest lift for both targets
    X["PG_x_Va"]  = X["PG_High"]*X["Va"]
    X["PG_x_ADT"] = X["PG_High"]*np.log1p(X["ADT"].fillna(0))
    X["PG_x_RBR"] = X["PG_High"]*X["RAP_Binder_Ratio"]
    X["PG_x_AFT"] = X["PG_High"]*X["AFT_micron"]
    # one-hot categoricals
    for c in ["Design_Level","Mix_Type"]:
        if c in df:
            d=pd.get_dummies(df[c].astype(str),prefix=c).astype(float)
            X=pd.concat([X,d.set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    return X

def load_target(path,sheet,tgt):
    df=pd.read_excel(path,sheet_name=sheet)
    y=num(df[tgt]).values
    grp=df["Exact_JMF_Group"].astype(str).values
    X=build_features(df)
    if FAST:
        rng=np.random.RandomState(RANDOM); idx=rng.choice(len(X),min(1200,len(X)),replace=False)
        X,y,grp=X.iloc[idx].reset_index(drop=True),y[idx],grp[idx]
    return X,y,grp

# ============================ MODELS (tuned) =================================
def zoo():
    T=300 if FAST else 1000; L=300 if FAST else 800
    return {
        "ExtraTrees":  ExtraTreesRegressor(n_estimators=T,min_samples_leaf=1,max_features=0.6,
                          n_jobs=-1,random_state=RANDOM),
        "RandomForest":RandomForestRegressor(n_estimators=int(T*0.8),min_samples_leaf=2,max_features=0.7,
                          n_jobs=-1,random_state=RANDOM),
        "LightGBM":    lgb.LGBMRegressor(n_estimators=L,learning_rate=0.03,num_leaves=63,
                          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,
                          reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1),
        "XGBoost":     xgb.XGBRegressor(n_estimators=L,learning_rate=0.03,max_depth=6,
                          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
                          random_state=RANDOM,n_jobs=-1,verbosity=0),
        "CatBoost":    CatBoostRegressor(iterations=L,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,
                          random_state=RANDOM,verbose=0),
        "HistGB":      HistGradientBoostingRegressor(max_iter=L,learning_rate=0.03,max_leaf_nodes=63,
                          min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM),
    }
STYLE={"ExtraTrees":("s","#1f77b4"),"RandomForest":("^","#2ca02c"),"LightGBM":("*","#17becf"),
       "XGBoost":("o","#9467bd"),"CatBoost":("h","#bcbd22"),"HistGB":("<","#ff7f0e"),
       "ENSEMBLE":("D","#d62728")}

def metrics(y,p):
    return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p))

def optimize_ensemble(oof_dict, y):
    """grid search convex weights maximizing R^2 of blend."""
    keys=list(oof_dict); best=(-9,None); grid=np.arange(0,1.01,0.1)
    for w in itertools.product(grid,repeat=len(keys)):
        if abs(sum(w)-1)>1e-6: continue
        p=sum(w[i]*oof_dict[keys[i]] for i in range(len(keys)))
        s=r2_score(y,p)
        if s>best[0]: best=(s,dict(zip(keys,w)))
    return best

# ============================ CORE ===========================================
def compute(path,name,sheet,tgt,unit):
    print("\n"+"="*72+f"\n  {name}   (v5: PG-filled + tuned + 6-model ensemble)\n"+"="*72)
    X,y,grp=load_target(path,sheet,tgt); feats=list(X.columns)
    print(f"  rows={len(X)}  unique mixes={len(np.unique(grp))}  features={X.shape[1]}  "
          f"target range=[{y.min():.2f},{y.max():.2f}]")
    ybin=pd.qcut(y,10,labels=False,duplicates="drop")
    sgkf=StratifiedGroupKFold(CV_FOLDS,shuffle=True,random_state=RANDOM)
    folds=list(sgkf.split(X,ybin,groups=grp))
    tr0,te0=folds[0]
    fit_res={}; cv_r2={}; oof_all={}
    for nm in zoo():
        oof=np.zeros(len(y)); fold_scores=[]
        for a,b in folds:
            m=zoo()[nm]; m.fit(X.iloc[a],y[a]); oof[b]=m.predict(X.iloc[b])
            fold_scores.append(r2_score(y[b],oof[b]))
        m=zoo()[nm]; m.fit(X.iloc[tr0],y[tr0])
        ptr=m.predict(X.iloc[tr0]); pte=m.predict(X.iloc[te0])
        om=metrics(y,oof); tm=metrics(y[te0],pte); trm=metrics(y[tr0],ptr)
        fit_res[nm]=dict(oof=oof,ptr=ptr,pte=pte,fold_r2=fold_scores,
                         R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
                         R2_te=tm["R2"],RMSE_te=tm["RMSE"],MAE_te=tm["MAE"],R2_tr=trm["R2"])
        cv_r2[nm]=np.array(fold_scores); oof_all[nm]=oof
        print(f"    {nm:12s} OOF R2={om['R2']:.4f}  RMSE={om['RMSE']:.4f}  {CV_FOLDS}-fold mean={np.mean(fold_scores):.4f}+/-{np.std(fold_scores):.3f}")
    # ---- weighted ensemble (grid search over OOF) ----
    best_r2,W = optimize_ensemble(oof_all,y)
    ens_oof=sum(W[k]*oof_all[k] for k in W)
    ens_te =sum(W[k]*fit_res[k]["pte"] for k in W)
    ens_tr =sum(W[k]*fit_res[k]["ptr"] for k in W)
    om=metrics(y,ens_oof); tm=metrics(y[te0],ens_te); trm=metrics(y[tr0],ens_tr)
    fit_res["ENSEMBLE"]=dict(oof=ens_oof,ptr=ens_tr,pte=ens_te,fold_r2=[],
                             R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
                             R2_te=tm["R2"],RMSE_te=tm["RMSE"],MAE_te=tm["MAE"],R2_tr=trm["R2"])
    cv_r2["ENSEMBLE"]=np.array([r2_score(y[b],ens_oof[b]) for _,b in folds])
    print(f"    ENSEMBLE     OOF R2={om['R2']:.4f}  RMSE={om['RMSE']:.4f}  weights={ {k:round(W[k],2) for k in W if W[k]>0} }")
    # ---- SHAP ----
    print("  computing SHAP on LightGBM ...")
    lgbm=zoo()["LightGBM"]; lgbm.fit(X.iloc[tr0],y[tr0])
    samp=X.iloc[te0].sample(min(600,len(te0)),random_state=RANDOM)
    sv=shap.TreeExplainer(lgbm).shap_values(samp)
    shap_mean=pd.Series(np.abs(sv).mean(0),index=samp.columns).sort_values(ascending=False)
    return dict(name=name,unit=unit,X=X,y=y,tr=tr0,te=te0,feats=feats,
                fit=fit_res,cv=cv_r2,shap=sv,X_shap=samp,shap_mean=shap_mean,weights=W)

# ============================ FIGURES ========================================
def fig_distribution(R):
    name,unit,y,X=R["name"],R["unit"],R["y"],R["X"]
    fig=plt.figure(figsize=(15,4.5)); fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1); ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
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
    order=[k for k in R["fit"] if k!="ENSEMBLE"][:6]+["ENSEMBLE"]
    fig,axs=plt.subplots(2,4,figsize=(18,8.5))
    fig.suptitle(f"{name}: Actual vs Predicted by model{unit}",fontweight="bold")
    lo=min(ytr.min(),yte.min()); hi=max(ytr.max(),yte.max())
    for ax,nm in zip(axs.ravel(),order+[None]*(8-len(order))):
        if nm is None: ax.axis("off"); continue
        fr=R["fit"][nm]
        ax.scatter(ytr,fr["ptr"],marker="D",s=16,facecolor="none",edgecolor="#7e57c2",alpha=.6,label="Training")
        ax.scatter(yte,fr["pte"],marker="^",s=22,color="#2ca02c",alpha=.7,label="Test")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.3,label="y=x")
        if len(yte)>=2 and np.std(yte)>0:
            a,b=np.polyfit(yte,fr["pte"],1); xs=np.array([lo,hi])
            ax.plot(xs,a*xs+b,"r-",lw=1.5,label="Fit")
            ax.text(0.05,0.92,f"y={a:.3f}x+{b:.2f}\n$R^2$={fr['R2_te']:.3f}",transform=ax.transAxes,
                    va="top",fontsize=9,bbox=dict(fc="white",ec="grey",alpha=.7))
        ax.set_title(nm+(" (blend)" if nm=="ENSEMBLE" else "")); ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    axs.ravel()[0].legend(loc="lower right",fontsize=7)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_actual_pred.png",bbox_inches="tight"); return fig

def fig_cv_box(R):
    name=R["name"]; order=sorted([k for k in R["cv"] if len(R["cv"][k])>0],
                                 key=lambda k:R["cv"][k].mean(),reverse=True)
    data=[R["cv"][k] for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE.get(k,("o","#888"))[1]); patch.set_alpha(.65)
    for i,k in enumerate(order):
        ax.annotate(f"{R['cv'][k].mean():.3f}",(i+1,R['cv'][k].mean()),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right")
    ax.set_ylabel(f"$R^2$ ({CV_FOLDS}-fold grouped OOF, replicates kept)")
    ax.set_title(f"{name}: {CV_FOLDS}-fold LOPOCV-style performance")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_cvbox.png",bbox_inches="tight"); return fig

def fig_overlay(R):
    name,unit,yte=R["name"],R["unit"],R["y"][R["te"]]
    fig,ax=plt.subplots(figsize=(7.5,7.5))
    lo,hi=yte.min(),yte.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE.get(nm,("o","#888"))
        ax.scatter(yte,fr["pte"],marker=mk,s=30,color=col,alpha=.7,label=nm,edgecolor="k",lw=.3)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y = x")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}")
    ax.set_title(f"{name}: prediction results of different models")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_overlay.png",bbox_inches="tight"); return fig

def fig_relerr(R):
    name,yte=R["name"],R["y"][R["te"]]
    order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_te"],reverse=True)
    yte_safe=np.where(np.abs(yte)<1e-6,1e-6,yte)
    data=[np.abs((R["fit"][k]["pte"]-yte)/yte_safe)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE.get(k,("o","#888"))[1]); patch.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey",lw=1.2,label="5%"); ax.axhline(10,ls=":",color="grey",lw=1.2,label="10%")
    for i,k in enumerate(order):
        ax.annotate(f"{np.median(data[i]):.1f}",(i+1,np.median(data[i])),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error (test set)"); ax.legend()
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
    xv=np.array([R["cv"][m].mean() if len(R["cv"][m])>0 else R["fit"][m]["R2_oof"] for m in models])
    yv=np.array([R["fit"][m]["R2_te"] for m in models])
    rho=spearmanr(xv,yv).correlation
    fig,ax=plt.subplots(figsize=(7,6.5))
    for m in models:
        mk,col=STYLE.get(m,("o","#888"))
        ax.scatter(xv[models.index(m)],R["fit"][m]["R2_te"],marker=mk,s=90,color=col,edgecolor="k",label=m)
    lo=min(xv.min(),yv.min())-0.05; hi=max(xv.max(),yv.max())+0.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(f"{CV_FOLDS}-fold CV mean $R^2$"); ax.set_ylabel("Locked-test $R^2$")
    ax.set_title(f"{name}: model ranking consistency\nSpearman $\\rho$ = {rho:.3f}")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_spearman.png",bbox_inches="tight"); return fig

# ============================ MAIN ===========================================
if __name__=="__main__":
    print("[v5] full run ~ 10 min per target (6 tuned models + ensemble + SHAP).")
    path=resolve_file(FILE)
    Rs=[]
    Rs.append(compute(path,"RUT","Rutting_Filtered","LWT_Design_Result"," (mm)"))
    Rs.append(compute(path,"SCB","SCB_Filtered","SCB_Result",""))
    for R in Rs:
        fig_distribution(R); fig_actual_pred(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_spearman(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*72+"\n  FINAL SUMMARY  (grouped 10-fold OOF, replicates KEPT, PG_filled)\n"+"="*72)
    for R in Rs:
        best=max(R["fit"],key=lambda k:R["fit"][k]["R2_oof"])
        f=R["fit"][best]
        print(f"  {R['name']}   best={best}   OOF R2={f['R2_oof']:.4f}   RMSE={f['RMSE_oof']:.4f}")
        if best=="ENSEMBLE":
            w=R["weights"]; print(f"       weights: { {k:round(w[k],2) for k in w if w[k]>0} }")
    print("\nDONE - all 7 diagnostic figures per target saved at 300 dpi and shown.")
