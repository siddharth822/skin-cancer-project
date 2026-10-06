import argparse, json, random, time
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import ManifestDataset, CLASS_NAMES
from model import create_model

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def tfms(train, size):
    items=[A.Resize(size,size)]
    if train:
        items += [A.HorizontalFlip(p=.5), A.Rotate(limit=20,p=.3),
                  A.RandomBrightnessContrast(p=.25), A.HueSaturationValue(p=.15)]
    items += [A.Normalize(), ToTensorV2()]
    return A.Compose(items)

def group_key(r):
    p=str(r.get("patient_id","")).strip()
    return p if p and p.lower()!="nan" else f"{r['source']}::{r['original_id']}"

def split_df(df):
    df=df[df.label.isin(CLASS_NAMES)].copy()
    df["_group"]=df.apply(group_key,axis=1)
    s=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    a,b=next(s.split(df,groups=df["_group"]))
    tr,va=df.iloc[a].copy(),df.iloc[b].copy()
    if set(tr["_group"]) & set(va["_group"]): raise RuntimeError("Group leakage detected")
    return tr,va

def sampler(df):
    k=df.source.astype(str)+"::"+df.label.astype(str)
    c=k.value_counts().to_dict()
    return WeightedRandomSampler([1/c[x] for x in k],len(k),replacement=True)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; loss_sum=0.; crit=nn.CrossEntropyLoss()
    for i,(x,y) in enumerate(loader,1):
        x,y=x.to(device),y.to(device)
        z=model(x); loss=crit(z,y); loss_sum += float(loss)*len(y)
        ys += y.cpu().tolist(); ps += z.argmax(1).cpu().tolist()
        if i%100==0 or i==len(loader): print(f"  validation {i}/{len(loader)}",flush=True)
    bal=balanced_accuracy_score(ys,ps)
    rep=classification_report(ys,ps,target_names=CLASS_NAMES,zero_division=0,output_dict=True)
    return loss_sum/len(loader.dataset),bal,rep

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default=r"D:\SkinCancerData\combined\combined_manifest.csv")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--size",type=int,default=160)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--output-dir",default=r"D:\SkinCancerData\models")
    p.add_argument("--from-scratch",action="store_true")
    p.add_argument("--fine-tune",action="store_true")
    a=p.parse_args(); seed_all()

    df=pd.read_csv(a.manifest); tr,va=split_df(df)
    print("\\nTRAIN distribution:"); print(tr.groupby(["source","label"]).size().unstack(fill_value=0))
    print("\\nVALIDATION distribution:"); print(va.groupby(["source","label"]).size().unstack(fill_value=0))

    trds=ManifestDataset(tr,tfms(True,a.size)); vads=ManifestDataset(va,tfms(False,a.size))
    trl=DataLoader(trds,batch_size=a.batch_size,sampler=sampler(tr),num_workers=a.workers)
    val=DataLoader(vads,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)

    device=torch.device("cpu")
    print("\\nDevice: cpu")
    print("Model: MobileNetV3-Small")
    print("Training samples:",len(trds),"Validation samples:",len(vads))
    print("Mode:","full fine-tune" if a.fine_tune else "classifier-head training",flush=True)

    model=create_model(pretrained=not a.from_scratch,freeze_backbone=not a.fine_tune).to(device)
    n=sum(x.numel() for x in model.parameters() if x.requires_grad)
    print("Trainable parameters:",f"{n:,}",flush=True)

    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss(label_smoothing=.05)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    best=-1.

    for epoch in range(1,a.epochs+1):
        start=time.time(); model.train(); run=0.
        for i,(x,y) in enumerate(trl,1):
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            z=model(x); loss=crit(z,y); loss.backward(); opt.step(); run += float(loss)*len(y)
            if i%100==0 or i==len(trl):
                print(f"  epoch {epoch}/{a.epochs} batch {i}/{len(trl)} | elapsed={(time.time()-start)/60:.1f} min",flush=True)
        vl,bal,rep=evaluate(model,val,device)
        tl=run/len(trds)
        print(f"Epoch {epoch}/{a.epochs} COMPLETE | train={tl:.4f} | val={vl:.4f} | balanced_acc={bal:.4f} | time={(time.time()-start)/60:.1f} min",flush=True)
        if bal>best:
            best=bal
            ck={"model_state":model.state_dict(),"class_names":CLASS_NAMES,"architecture":"mobilenet_v3_small",
                "image_size":a.size,"balanced_accuracy":bal,"validation_report":rep,
                "dataset":"PAD-UFES-20 + ISIC-2019",
                "training_mode":"from_scratch" if a.from_scratch else ("imagenet_finetune" if a.fine_tune else "imagenet_head_only")}
            torch.save(ck,out/"skinsight_mobilenetv3_small.pt")
            (out/"metrics.json").write_text(json.dumps({"best_balanced_accuracy":bal,"report":rep},indent=2),encoding="utf-8")
            print("Saved:",out/"skinsight_mobilenetv3_small.pt",flush=True)
    print("Training complete. Best balanced accuracy:",best)

if __name__=="__main__": main()
