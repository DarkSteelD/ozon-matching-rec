from __future__ import annotations
import argparse,json,re,time
from pathlib import Path
import numpy as np,polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,log_loss,roc_auc_score,average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from transformers import AutoTokenizer

TOK=re.compile(r"\w+",re.UNICODE)
def ws(x):return set(TOK.findall((x or "").lower()))
def features(d):
 out=[]
 for n1,n2,a1,a2,c in zip(d['name1'],d['name2'],d['attrs1'],d['attrs2'],d['category']):
  x,y=ws(n1),ws(n2);u=len(x|y); l=[len(n1 or ''),len(n2 or ''),len(a1 or ''),len(a2 or '')]
  out.append([c,*l,abs(l[0]-l[1]),abs(l[2]-l[3]),min(l[0],l[1])/(max(l[0],l[1])+1),min(l[2],l[3])/(max(l[2],l[3])+1),len(x&y)/u if u else 0,float((n1 or '').strip().lower()==(n2 or '').strip().lower()),float(not a1),float(not a2),sum(ch.isdigit() for ch in (n1 or '')+(a1 or '')),sum(ch.isdigit() for ch in (n2 or '')+(a2 or ''))])
 return np.asarray(out,dtype=object)
def ap(y,p,cats):return float(np.mean([average_precision_score(y[cats==c],p[cats==c]) for c in sorted(set(cats))]))
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--preds',required=True);a.add_argument('--output',required=True);a.add_argument('--minilm',required=True);a.add_argument('--e5',required=True);z=a.parse_args();root=Path(z.output);root.mkdir(parents=True,exist_ok=True)
 d=pl.read_parquet(z.data);rng=np.random.default_rng(20260824);idx=np.sort(rng.choice(d.height,120000,replace=False));s=d[idx];X=features(s);yy=np.array([int(x[-2:])-1 for x in s['fold']]);perm=rng.permutation(len(yy));cut=90000;tr,te=perm[:cut],perm[cut:]
 pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),[0]),('num',StandardScaler(),list(range(1,X.shape[1])))])
 clf=make_pipeline(pre,LogisticRegression(max_iter=200,C=.3,n_jobs=8));t=time.time();clf.fit(X[tr],yy[tr]);prob=clf.predict_proba(X[te]);pred=prob.argmax(1)
 foldclf={'rows':len(yy),'balanced_accuracy':float(balanced_accuracy_score(yy[te],pred)),'macro_ovr_auc':float(roc_auc_score(yy[te],prob,multi_class='ovr',average='macro')),'log_loss':float(log_loss(yy[te],prob)),'chance_log_loss':float(np.log(4)),'seconds':time.time()-t}
 # Existing architecture delta stability across all four OOF folds/categories.
 truth=d.select('fold','id1','id2','target','category');models={}
 for name in ['ce_rubase_e2_len384','ce_e5_len288','ce_mdeb_len224']:
  parts=[]
  for p in sorted(Path(z.preds,name).glob('fold_*.csv')):parts.append(pl.read_csv(p))
  models[name]=pl.concat(parts).rename({'predict':name})
 joined=truth
 for name,p in models.items():joined=joined.join(p,on=['id1','id2'],validate='1:1')
 base='ce_rubase_e2_len384';delta_stability={}
 for name in ['ce_e5_len288','ce_mdeb_len224']:
  matrix=[]
  for fold in sorted(set(joined['fold'])):
   q=joined.filter(pl.col('fold')==fold);y=q['target'].to_numpy();cats=q['category'].to_numpy();v=[]
   for c in sorted(set(cats)):
    m=cats==c;v.append(average_precision_score(y[m],q[name].to_numpy()[m])-average_precision_score(y[m],q[base].to_numpy()[m]))
   matrix.append(v)
  M=np.asarray(matrix);cors=[];sign=[]
  for i in range(4):
   for j in range(i+1,4):cors.append(float(np.corrcoef(M[i],M[j])[0,1]));sign.append(float(np.mean(np.sign(M[i])==np.sign(M[j]))))
  delta_stability[name]={'fold_category_deltas':M.tolist(),'pairwise_correlation_mean':float(np.mean(cors)),'pairwise_correlation_min':float(np.min(cors)),'pairwise_sign_agreement_mean':float(np.mean(sign)),'per_category_delta_sd_mean':float(M.std(0).mean())}
 # Tokenizer fertility and exact vocabulary provenance.
 sample=d[np.sort(rng.choice(d.height,20000,replace=False))]
 texts=[f"{n1} | {c} | {a1 or ''}" for n1,a1,c in zip(sample['name1'],sample['attrs1'],sample['category'])]
 fert={}
 for name,path in [('minilm',z.minilm),('e5',z.e5)]:
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True);enc=tok(texts,truncation=False,add_special_tokens=True);lens=np.array([len(x) for x in enc['input_ids']]);fert[name]={'mean':float(lens.mean()),'p50':float(np.quantile(lens,.5)),'p90':float(np.quantile(lens,.9)),'p99':float(np.quantile(lens,.99)),'over160':float(np.mean(lens>160)),'over224':float(np.mean(lens>224)),'vocab_size':len(tok)}
 result={'fold_classifier':foldclf,'architecture_delta_stability':delta_stability,'tokenizer_fertility':fert};(root/'cpu_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
