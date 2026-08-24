from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

def ap(y,p): return float(average_precision_score(y,p))

def main():
    a=argparse.ArgumentParser(); a.add_argument("--data",required=True); a.add_argument("--masks",required=True)
    a.add_argument("--pred-root",required=True); a.add_argument("--folds",default="fold_01,fold_02"); a.add_argument("--output",required=True); args=a.parse_args()
    d=pl.read_parquet(args.data).join(pl.read_parquet(args.masks),on=["id1","id2"],validate="1:1")
    rows=[]
    for fold in args.folds.split(","):
        truth=d.filter(pl.col("fold")==fold)
        scores={}
        for vd in sorted(Path(args.pred_root).iterdir()):
            p=pl.read_csv(vd/f"{fold}.csv").rename({"predict":vd.name})
            truth=truth.join(p,on=["id1","id2"],validate="1:1"); scores[vd.name]=vd.name
        y=truth["target"].to_numpy(); cats=truth["category"].to_numpy()
        base={}
        for variant in scores:
            pred=truth[variant].to_numpy(); bycat={}
            for cat in sorted(set(cats)):
                m=cats==cat; bycat[cat]=ap(y[m],pred[m])
            slices={}
            for col in [c for c in truth.columns if c.startswith("unit_")]:
                m=truth[col].to_numpy()
                slices[col]={"rows":int(m.sum()),"positives":int(y[m].sum()),"prauc":ap(y[m],pred[m]) if y[m].sum() else None}
            row={"variant":variant,"fold":fold,"prauc":ap(y,pred),"macro_category_prauc":float(np.mean(list(bycat.values()))),"per_category":bycat,"slices":slices}
            if variant=="baseline": base=row
            rows.append(row)
        for row in rows:
            if row["fold"]!=fold or row["variant"]=="baseline": continue
            row["delta_prauc"]=row["prauc"]-base["prauc"]; row["delta_macro"]=row["macro_category_prauc"]-base["macro_category_prauc"]
            row["per_category_delta"]={k:row["per_category"][k]-base["per_category"][k] for k in base["per_category"]}
            row["slice_delta"]={k:row["slices"][k]["prauc"]-base["slices"][k]["prauc"] for k in base["slices"] if row["slices"][k]["prauc"] is not None}
        base["delta_prauc"]=base["delta_macro"]=0.0
    ag=[]
    for v in sorted(set(r["variant"] for r in rows)):
        z=[r for r in rows if r["variant"]==v]
        ag.append({"variant":v,"folds":len(z),"prauc_mean":float(np.mean([r["prauc"] for r in z])),"prauc_std":float(np.std([r["prauc"] for r in z],ddof=1)) if len(z)>1 else None,"delta_prauc_mean":float(np.mean([r["delta_prauc"] for r in z])),"macro_mean":float(np.mean([r["macro_category_prauc"] for r in z])),"delta_macro_mean":float(np.mean([r["delta_macro"] for r in z])),"same_positive_sign":all(r["delta_prauc"]>0 for r in z),"worst_category_delta":min(min(r.get("per_category_delta",{"_":0}).values()) for r in z)})
    out={"rows":rows,"aggregates":ag}; Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    for x in ag: print(x)
if __name__=="__main__": main()
