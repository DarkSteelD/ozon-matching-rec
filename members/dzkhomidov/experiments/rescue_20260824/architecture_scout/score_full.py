from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

def ranks(x,cats):
    out=np.empty(len(x),np.float64)
    for c in np.unique(cats):
        ix=np.flatnonzero(cats==c);order=np.argsort(x[ix],kind="stable")
        rr=np.empty(len(ix),np.float64);rr[order]=(np.arange(len(ix))+.5)/len(ix);out[ix]=rr
    return out

def metrics(y,p,cats):
    by={str(c):float(average_precision_score(y[cats==c],p[cats==c])) for c in np.unique(cats)}
    return {"pooled_ap":float(average_precision_score(y,p)),"macro_category_ap":float(np.mean(list(by.values()))),"by_category":by}

def main():
    p=argparse.ArgumentParser();p.add_argument("--hard",required=True);p.add_argument("--baseline",required=True);p.add_argument("--candidate",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    hard=pl.read_parquet(a.hard);result={"weight":0.1,"shuffle_seed_base":20260824,"folds":{}}
    for fold in ("fold_01","fold_02"):
        d=hard.filter(pl.col("fold")==fold).select("id1","id2","target","category")
        b=pl.read_csv(Path(a.baseline)/f"{fold}.csv").rename({"predict":"baseline"});m=pl.read_csv(Path(a.candidate)/f"{fold}.csv").rename({"predict":"minilm"})
        d=d.join(b,on=["id1","id2"],how="left",validate="1:1").join(m,on=["id1","id2"],how="left",validate="1:1");assert d["baseline"].null_count()==d["minilm"].null_count()==0
        y=d["target"].to_numpy();cats=d["category"].to_numpy();bp=d["baseline"].to_numpy();mp=d["minilm"].to_numpy();br=ranks(bp,cats);mr=ranks(mp,cats)
        shuffled=np.empty(len(mr));rng=np.random.default_rng(20260824+int(fold[-2:]))
        for c in np.unique(cats):
            ix=np.flatnonzero(cats==c);shuffled[ix]=mr[rng.permutation(ix)]
        variants={"baseline":bp,"minilm_standalone":mp,"pretrained10":.9*br+.1*mr,"shuffled_pretrained10":.9*br+.1*shuffled};base=metrics(y,bp,cats);row={"rows":len(y),"variants":{}}
        for name,sc in variants.items():
            z=metrics(y,sc,cats);z["delta_macro_vs_baseline"]=z["macro_category_ap"]-base["macro_category_ap"];z["delta_pooled_vs_baseline"]=z["pooled_ap"]-base["pooled_ap"];row["variants"][name]=z
        result["folds"][fold]=row
    ds=[result["folds"][f]["variants"]["pretrained10"]["delta_macro_vs_baseline"] for f in ("fold_01","fold_02")]
    result["gate"]={"threshold":0.001,"fold_deltas":ds,"pass":bool(all(x>.001 for x in ds))}
    Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(result["gate"],indent=2))
if __name__=="__main__":main()
