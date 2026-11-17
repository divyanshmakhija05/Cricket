"""EVIDENCE E1: compares win-probability models (logistic variants vs gradient boosting) with cross-validation grouped by match.
Usage: python E1_model_selection.py <out_dir>      (reads <out_dir>/tables/ball_state.csv.gz, writes <out_dir>/logs/E1_model_selection.log)"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
OUTROOT = sys.argv[1] if len(sys.argv) > 1 else "output"; os.makedirs(OUTROOT + "/logs", exist_ok=True)
class _Tee:
    def __init__(s, f): s.f = f; s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()
sys.stdout = _Tee(open(OUTROOT + "/logs/E1_model_selection.log", "w"))
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold   # (no longer used: see match_folds)
def match_folds(groups, n=5):
    """Deterministic, library-independent cross-validation folds: a match always goes to fold (match_id mod n),
    so every ball of a match stays together and results are identical on any machine / library version."""
    f = np.asarray(groups) % n
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(n)]
from sklearn.metrics import log_loss, roc_auc_score
b=pd.read_csv(OUTROOT+'/tables/ball_state.csv.gz',low_memory=False)
n=b.groupby('match_id').innings.nunique()
k=b[b.comp_group.isin(['IPL','T20I'])&~b.no_result&~b.dls&~b.tied&b.match_id.isin(n[n==2].index)&b.batting_won.notna()].reset_index(drop=True)
k['ipl']=(k.comp_group=='IPL')*1.0
def cv(X,y,g,mk):
    p=np.zeros(len(y))
    for tr,te in match_folds(g): p[te]=mk().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return round(log_loss(y,np.clip(p,1e-4,1-1e-4)),4), round(roc_auc_score(y,p),4)
d=k[k.innings==2]; y=d.batting_won.values; g=d.match_id.values
bl,wk,sc,tg=d.balls_left.values,d.wkts_in_hand.values,d.score_before.values,d.target.values
rn=np.clip(tg-sc,1,None); rrr=np.clip(rn/np.maximum(bl,1)*6,0,36)
lr=lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0))
F={}
F['base']=np.column_stack([bl,wk,rn,rrr,tg,d.ipl])
F['eng']=np.column_stack([bl,wk,rn,rrr,np.log(rn),np.sqrt(bl),bl*wk,rn/(wk+1),rrr*wk,np.log1p(rrr),d.ipl,tg])
F['eng2']=np.column_stack([F['eng'],np.maximum(rn-bl*2.2,0),rrr**2])
print("INNINGS 2 (logloss, auc)")
for nm,X in F.items(): print(" LR",nm,cv(X,y,g,lr))
sp=lambda: make_pipeline(StandardScaler(),PolynomialFeatures(2),StandardScaler(),LogisticRegression(max_iter=2000,C=0.1))
print(" LR poly2 on eng",cv(F['eng'],y,g,sp))
print(" GBM reg.",cv(F['base'],y,g,lambda:HGB(max_iter=120,learning_rate=0.03,max_leaf_nodes=6,min_samples_leaf=1500,l2_regularization=5,random_state=0)))
d=k[k.innings==1]; y=d.batting_won.values; g=d.match_id.values
bl,wk,sc,lb=d.balls_left.values,d.wkts_in_hand.values,d.score_before.values,d.legal_balls_before.values
crr=np.where(lb>0,sc/np.maximum(lb,1)*6,8.0)
proj=sc+bl*crr/6
G={}
G['base']=np.column_stack([bl,wk,sc,crr,d.ipl])
G['eng']=np.column_stack([bl,wk,sc,crr,proj,wk*bl,sc*wk,np.sqrt(bl),d.ipl])
print("INNINGS 1")
for nm,X in G.items(): print(" LR",nm,cv(X,y,g,lr))
print(" LR poly2",cv(G['base'],y,g,sp))
print(" GBM reg.",cv(G['base'],y,g,lambda:HGB(max_iter=120,learning_rate=0.03,max_leaf_nodes=6,min_samples_leaf=1500,l2_regularization=5,random_state=0)))
