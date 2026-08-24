import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def ece(y, p, bins=15):
    edge = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edge)-1, 0, bins-1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hard',required=True); ap.add_argument('--preds',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--variants',required=True); ap.add_argument('--folds',required=True)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(a.hard,columns=['fold','id1','id2','target','category'])
    rows=[]; cats=[]
    for fold in a.folds.split(','):
        ref=d[d.fold.eq(fold)].reset_index(drop=True); y=ref.target.to_numpy()
        for variant in a.variants.split(','):
            p=pd.read_csv(Path(a.preds)/variant/f'{fold}.csv')
            assert np.array_equal(p[['id1','id2']].to_numpy(),ref[['id1','id2']].to_numpy())
            pred=p.predict.to_numpy(); aps=[]
            for cat,g in ref.assign(pred=pred).groupby('category',sort=True):
                v=average_precision_score(g.target,g.pred); aps.append(v)
                cats.append({'fold':fold,'variant':variant,'category':cat,'rows':len(g),'ap':v})
            rows.append({'fold':fold,'variant':variant,'macro_category_ap':np.mean(aps),
                         'pooled_ap':average_precision_score(y,pred),'brier':brier_score_loss(y,pred),
                         'logloss':log_loss(y,pred),'ece15':ece(y,pred)})
    m=pd.DataFrame(rows); base=m[m.variant.eq('bce')].set_index('fold')
    for col in ['macro_category_ap','pooled_ap','brier','logloss','ece15']:
        m['delta_'+col]=[getattr(r,col)-base.loc[r.fold,col] for r in m.itertuples()]
    gate=m[m.variant.eq('poly05')].set_index('fold').delta_macro_category_ap.gt(.001).all()
    c=pd.DataFrame(cats); q=c.pivot(index=['fold','category'],columns='variant',values='ap').dropna()
    delta=(q.poly05-q.bce).groupby('category').mean().to_numpy(); rng=np.random.default_rng(20260824)
    boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
    result={'gate_threshold_each_fold':.001,'poly05_gate_pass':bool(gate),
            'bootstrap95_category_delta':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'metrics':m.to_dict('records')}
    m.to_csv(out/'metrics.csv',index=False); c.to_csv(out/'category_metrics.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(m.to_string(index=False)); print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
