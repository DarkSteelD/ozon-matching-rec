from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

def subset(df, seed, heldout):
    rng=np.random.default_rng(seed); picked=[]
    train=df.with_row_index("row").filter(pl.col("fold")!=heldout)
    for _,g in train.group_by("category",maintain_order=True):
        rows=g["row"].to_numpy(); y=g["target"].to_numpy(); n=2000
        pos=rows[y==1]; neg=rows[y==0]; npick=round(n*len(pos)/len(rows))
        picked.extend(rng.choice(pos,npick,replace=False)); picked.extend(rng.choice(neg,n-npick,replace=False))
    picked=np.asarray(sorted(picked)); assert len(picked)==40000
    return picked

def tokens(df,tok,max_len):
    def text(n,a,c): return f"{n} | {c} | {a}" if a else f"{n} | {c}"
    a=[text(*x) for x in zip(df["name1"],df["attrs1"],df["category"])]
    b=[text(*x) for x in zip(df["name2"],df["attrs2"],df["category"])]
    ids=np.zeros((df.height,max_len),np.int32); tt=np.zeros((df.height,max_len),np.uint8)
    for s in range(0,df.height,20000):
        e=min(s+20000,df.height); z=tok(a[s:e],b[s:e],truncation=True,max_length=max_len,padding="max_length",return_tensors="np")
        ids[s:e]=z["input_ids"].astype(np.int32)
        if "token_type_ids" in z: tt[s:e]=z["token_type_ids"].astype(np.uint8)
    return ids,tt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--rubase",required=True);p.add_argument("--minilm",required=True);p.add_argument("--steps",type=int,default=500)
    p.add_argument("--bs",type=int,default=128);p.add_argument("--max-len",type=int,default=160);p.add_argument("--seed",type=int,default=20260814)
    p.add_argument("--fold",default="fold_01");a=p.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"]="true";torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    df=pl.read_parquet(a.data); tr=subset(df,a.seed,a.fold); ev=np.flatnonzero(df["fold"].to_numpy()==a.fold); y=df["target"].to_numpy().astype(np.float32)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True); manifest={"args":vars(a),"train_rows":tr.tolist(),"runs":[]}
    for variant,path,random_init in [("rubase",a.rubase,False),("minilm",a.minilm,False),("minilm_random",a.minilm,True)]:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True); t=time.time();ids,tt=tokens(df,tok,a.max_len);toksec=time.time()-t
        if random_init:
            cfg=AutoConfig.from_pretrained(path,local_files_only=True);cfg.num_labels=1;model=AutoModelForSequenceClassification.from_config(cfg)
        else: model=AutoModelForSequenceClassification.from_pretrained(path,num_labels=1,ignore_mismatched_sizes=True,local_files_only=True)
        model=model.cuda(); torch.cuda.reset_peak_memory_stats(); use_tt=getattr(model.config,"type_vocab_size",0)>1
        opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-5,total_steps=a.steps,pct_start=.06,anneal_strategy="linear")
        lossf=torch.nn.BCEWithLogitsLoss();rng=np.random.default_rng(a.seed);order=[]
        while len(order)<a.steps*a.bs: order.extend(rng.permutation(tr).tolist())
        started=time.time();peak=0;model.train()
        for step in range(a.steps):
            idx=np.asarray(sorted(order[step*a.bs:(step+1)*a.bs]));bi=torch.from_numpy(ids[idx].astype(np.int64)).cuda();bt=torch.from_numpy(tt[idx].astype(np.int64)).cuda();by=torch.from_numpy(y[idx]).cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16): z=model(input_ids=bi,attention_mask=(bi!=tok.pad_token_id).long(),token_type_ids=bt if use_tt else None).logits.squeeze(-1);loss=lossf(z,by)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step();opt.zero_grad(set_to_none=True)
            peak=max(peak,torch.cuda.max_memory_allocated());
            if (step+1)%100==0: print(variant,step+1,loss.item(),flush=True)
        trainsec=time.time()-started;model.eval();out=np.zeros(len(ev),np.float32);estart=time.time()
        with torch.no_grad():
            for s in range(0,len(ev),a.bs*4):
                idx=ev[s:s+a.bs*4];bi=torch.from_numpy(ids[idx].astype(np.int64)).cuda();bt=torch.from_numpy(tt[idx].astype(np.int64)).cuda()
                with torch.autocast("cuda",dtype=torch.bfloat16): z=model(input_ids=bi,attention_mask=(bi!=tok.pad_token_id).long(),token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                out[s:s+len(idx)]=torch.sigmoid(z.float()).cpu().numpy()
        evalsec=time.time()-estart;od=root/"preds"/variant;od.mkdir(parents=True,exist_ok=True);pl.DataFrame({"id1":df["id1"][ev],"id2":df["id2"][ev],"predict":out}).write_csv(od/f"{a.fold}.csv")
        manifest["runs"].append({"variant":variant,"tokenize_seconds":toksec,"train_seconds":trainsec,"eval_seconds":evalsec,"train_examples_per_second":a.steps*a.bs/trainsec,"eval_examples_per_second":len(ev)/evalsec,"peak_memory_bytes":peak})
        (root/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");del model,ids,tt;torch.cuda.empty_cache()
if __name__=="__main__":main()
