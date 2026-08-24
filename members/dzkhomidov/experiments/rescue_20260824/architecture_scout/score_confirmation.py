from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

def rank_by_category(scores, categories):
    out=np.empty(len(scores),np.float64)
    for cat in np.unique(categories):
        ix=np.flatnonzero(categories==cat)
        order=np.argsort(scores[ix],kind="stable")
        ranks=np.empty(len(ix),np.float64);ranks[order]=(np.arange(len(ix))+.5)/len(ix)
        out[ix]=ranks
    return out

def macro_ap(y, scores, categories):
    vals={str(cat):float(average_precision_score(y[categories==cat],scores[categories==cat])) for cat in np.unique(categories)}
    return float(np.mean(list(vals.values()))),vals

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--pred-root",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    df=pl.read_parquet(a.data);root=Path(a.pred_root);result={"weight":0.1,"shuffle_seed_base":20260824,"folds":{}}
    for fold in ("fold_01","fold_02"):
        ev=df.filter(pl.col("fold")==fold).select("id1","id2","target","category")
        scores={}
        for arm in ("rubase","minilm","minilm_random"):
            pred=pl.read_csv(root/arm/f"{fold}.csv").rename({"predict":arm})
            ev=ev.join(pred,on=["id1","id2"],how="left")
            assert ev[arm].null_count()==0
            scores[arm]=ev[arm].to_numpy()
        y=ev["target"].to_numpy();cats=ev["category"].to_numpy();rr=rank_by_category(scores["rubase"],cats);mr=rank_by_category(scores["minilm"],cats);nr=rank_by_category(scores["minilm_random"],cats)
        shuffled=np.empty(len(mr),np.float64);rng=np.random.default_rng(20260824+int(fold[-2:]))
        for cat in np.unique(cats):
            ix=np.flatnonzero(cats==cat);shuffled[ix]=mr[rng.permutation(ix)]
        variants={"baseline":scores["rubase"],"pretrained10":.9*rr+.1*mr,"random_init10":.9*rr+.1*nr,"shuffled_pretrained10":.9*rr+.1*shuffled}
        base,_=macro_ap(y,scores["rubase"],cats);fr={"rows":len(y),"baseline_macro_ap":base,"variants":{}}
        for name,sc in variants.items():
            macro,bycat=macro_ap(y,sc,cats);fr["variants"][name]={"macro_ap":macro,"delta":macro-base,"by_category":bycat}
        result["folds"][fold]=fr
    p1=result["folds"]["fold_01"]["variants"]["pretrained10"]["delta"];p2=result["folds"]["fold_02"]["variants"]["pretrained10"]["delta"]
    gaps=[]
    for fold in ("fold_01","fold_02"):
        v=result["folds"][fold]["variants"];gaps.extend([v["pretrained10"]["delta"]-v["random_init10"]["delta"],v["pretrained10"]["delta"]-v["shuffled_pretrained10"]["delta"]])
    result["gate"]={"pretrained_gt_0.001_both":bool(p1>.001 and p2>.001),"controls_lower_by_0.001_each_fold":bool(min(gaps)>=.001),"pass":bool(p1>.001 and p2>.001 and min(gaps)>=.001),"control_gaps":gaps}
    Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(result["gate"],indent=2))
if __name__=="__main__":main()
