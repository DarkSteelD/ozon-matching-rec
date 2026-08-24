import argparse,json
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score

def ranks(x,cats):
 out=np.zeros(len(x));
 for c in set(cats):
  m=cats==c;v=x[m];o=np.argsort(v,kind="mergesort");q=np.empty(len(v));q[o]=np.arange(len(v));out[m]=(q+.5)/len(v)
 return out
def metrics(y,p,cats):
 pc={c:float(average_precision_score(y[cats==c],p[cats==c])) for c in sorted(set(cats))}
 return {"pooled":float(average_precision_score(y,p)),"macro":float(np.mean(list(pc.values()))),"per_category":pc}
def main():
 a=argparse.ArgumentParser();a.add_argument("--data",required=True);a.add_argument("--pred-root",required=True);a.add_argument("--output",required=True);z=a.parse_args()
 d=pl.read_parquet(z.data).filter(pl.col("fold")=="fold_01");pred={}
 for p in Path(z.pred_root).glob("*/fold_01.csv"):
  q=pl.read_csv(p).rename({"predict":p.parent.name});d=d.join(q,on=["id1","id2"],validate="1:1");pred[p.parent.name]=d[p.parent.name].to_numpy()
 y=d["target"].to_numpy();cats=d["category"].to_numpy();base=metrics(y,pred["rubase"],cats);rows={k:metrics(y,v,cats) for k,v in pred.items()}
 br=ranks(pred["rubase"],cats);cr=ranks(pred["minilm"],cats)
 for w in [.1,.25,.5]: rows[f"blend_{w}"]=metrics(y,(1-w)*br+w*cr,cats)
 for k,v in rows.items():v["delta_macro_vs_rubase"]=v["macro"]-base["macro"];v["delta_pooled_vs_rubase"]=v["pooled"]-base["pooled"]
 Path(z.output).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
 for k,v in rows.items():print(k,v["pooled"],v["macro"],v["delta_macro_vs_rubase"])
if __name__=="__main__":main()
