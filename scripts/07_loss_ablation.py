"""Optional extra credit: trains BCE, Dice and BCE+Dice U-Nets and compares best validation Dice."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd, torch
from torch.utils.data import DataLoader
from nuclei_pipeline.config import DATA_DIR,METRIC_DIR,MODEL_DIR,SEED
from nuclei_pipeline.data import NucleiDataset,seed_everything
from nuclei_pipeline.unet import SmallUNet,make_loss
from nuclei_pipeline.train_utils import pick_device,run_epoch

parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=10); args=parser.parse_args()
device=pick_device(); results=[]
for loss_name in ["bce","dice","bce+dice"]:
    seed_everything(SEED); tr=DataLoader(NucleiDataset(DATA_DIR,"train"),batch_size=4,shuffle=True); va=DataLoader(NucleiDataset(DATA_DIR,"val"),batch_size=4)
    m=SmallUNet().to(device); c=make_loss(loss_name); o=torch.optim.Adam(m.parameters(),lr=1e-3); best=0
    for e in range(1,args.epochs+1):
        run_epoch(m,tr,c,o,device,True); vl,vd,vi=run_epoch(m,va,c,o,device,False); best=max(best,vd); print(loss_name,e,vd)
    results.append({"loss":loss_name,"best_val_dice":best,"last_val_iou":vi})
pd.DataFrame(results).to_csv(METRIC_DIR/"loss_ablation.csv",index=False); print(pd.DataFrame(results))
