#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT=Path('/home/dzkhomidov/matching-work/rescue_20260824/residual_matrix')
MATRIX=Path('/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet')
BASE=Path('/home/dzkhomidov/matching-work/rescue_20260824/category_blend/artifacts_v2/heldout_predictions.parquet')
GRID=np.array([(a,b) for a in [0,.025,.05,.1,.2] for b in [0,.025,.05,.1,.2] if a+b<=.35])

def ap(y,p): return float(average_precision_score(y,p))
def ranks(df,x):
 out=np.empty(len(df),np.float32)
 for idx in df.groupby(['fold','category'],sort=False).indices.values():
  idx=np.asarray(idx);out[idx]=pd.Series(x[idx]).rank(method='average',pct=True).to_numpy(np.float32)
 return out
def macro(df,p): return float(np.mean([ap(g.target,p[g.index]) for _,g in df.groupby('category',sort=True)]))

def arm(df,base,x1,x2):
 folds=sorted(df.fold.unique());cats=sorted(df.category.unique());out=np.empty(len(df),np.float32);chosen=[]
 allp=np.stack([(1-a-b)*base+a*x1+b*x2 for a,b in GRID],axis=1)
 for held in folds:
  train=df.fold.ne(held).to_numpy();val=df.fold.eq(held).to_numpy()
  gs=np.array([np.mean([ap(df.target[train&df.category.eq(c)],allp[train&df.category.eq(c),j]) for c in cats]) for j in range(len(GRID))])
  gi=int(np.argmax(gs));gw=GRID[gi];wv=np.empty((val.sum(),2),np.float32);vdf=df.loc[val]
  for c in cats:
   tr=train&df.category.eq(c).to_numpy();scores=np.array([ap(df.target[tr],allp[tr,j]) for j in range(len(GRID))])
   best=np.flatnonzero(np.isclose(scores,scores.max(),atol=1e-12,rtol=0));ci=best[np.argmin(np.sum((GRID[best]-gw)**2,axis=1))]
   cw=GRID[ci];sw=.75*cw+.25*gw;wv[vdf.category.eq(c).to_numpy()]=sw
   chosen.append({'held_fold':held,'category':c,'global_w1':gw[0],'global_w2':gw[1],'cat_w1':cw[0],'cat_w2':cw[1],'shrink_w1':sw[0],'shrink_w2':sw[1]})
  out[val]=(1-wv.sum(1))*base[val]+wv[:,0]*x1[val]+wv[:,1]*x2[val]
 return out,chosen

df=pd.read_parquet(MATRIX);b=pd.read_parquet(BASE)
assert np.array_equal(df[['fold','target','category']].to_numpy(),b[['fold','target','category']].to_numpy())
base=ranks(df,b.category_grid_shrink75.to_numpy());x1=ranks(df,df.final_stack_all.to_numpy());x2=ranks(df,df.ce_final_combo.to_numpy())
pred,chosen=arm(df,base,x1,x2)
rng=np.random.default_rng(20260824)
sx=[]
for x in [x1,x2]:
 z=x.copy()
 for idx in df.groupby(['fold','category'],sort=False).indices.values(): idx=np.asarray(idx);z[idx]=z[rng.permutation(idx)]
 sx.append(z)
ctrl,_=arm(df,base,*sx)
rows=[]
for name,p in [('pair',pred),('shuffled_pair',ctrl)]:
 row={'variant':name,'macro':macro(df,p),'delta':macro(df,p)-macro(df,base)}
 for f in sorted(df.fold.unique()):
  m=df.fold.eq(f).to_numpy();sub=df.loc[m].reset_index(drop=True);row[f'delta_{f}']=macro(sub,p[m])-macro(sub,base[m])
 rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'pair_metrics.csv',index=False);pd.DataFrame(chosen).to_csv(ROOT/'pair_weights.csv',index=False)
print(json.dumps(rows,indent=2))
