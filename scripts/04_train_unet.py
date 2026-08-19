import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from nuclei_pipeline.config import DATA_DIR, MODEL_DIR, METRIC_DIR, FIGURE_DIR, SEED
from nuclei_pipeline.data import NucleiDataset, seed_everything
from nuclei_pipeline.unet import SmallUNet, make_loss
from nuclei_pipeline.train_utils import pick_device, run_epoch, save_history

parser=argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=20)
parser.add_argument("--batch-size",type=int,default=4)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--loss",default="bce+dice",choices=["bce","dice","bce+dice"])
args=parser.parse_args()
seed_everything(SEED)
device=pick_device(); print("Device:",device)
train_loader=DataLoader(NucleiDataset(DATA_DIR,"train"),batch_size=args.batch_size,shuffle=True,num_workers=0)
val_loader=DataLoader(NucleiDataset(DATA_DIR,"val"),batch_size=args.batch_size,shuffle=False,num_workers=0)
model=SmallUNet().to(device); criterion=make_loss(args.loss); optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
history=[]; best=-1
for epoch in range(1,args.epochs+1):
    tr_loss,tr_dice,tr_iou=run_epoch(model,train_loader,criterion,optimizer,device,True)
    va_loss,va_dice,va_iou=run_epoch(model,val_loader,criterion,optimizer,device,False)
    row={"epoch":epoch,"train_loss":tr_loss,"train_dice":tr_dice,"train_iou":tr_iou,"val_loss":va_loss,"val_dice":va_dice,"val_iou":va_iou}
    history.append(row)
    print(f"Epoch {epoch:02d}/{args.epochs} loss={tr_loss:.4f} val_loss={va_loss:.4f} val_dice={va_dice:.4f} val_iou={va_iou:.4f}")
    if va_dice>best:
        best=va_dice; torch.save({"model_state":model.state_dict(),"loss":args.loss,"epoch":epoch,"val_dice":va_dice,"val_iou":va_iou},MODEL_DIR / "unet_best.pt")
save_history(history,METRIC_DIR / "unet_history.csv")
df=pd.DataFrame(history)
plt.figure(figsize=(7,4)); plt.plot(df.epoch,df.train_loss,label="train"); plt.plot(df.epoch,df.val_loss,label="validation"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_DIR/"unet_loss_curve.png",dpi=180); plt.close()
plt.figure(figsize=(7,4)); plt.plot(df.epoch,df.val_dice,label="Validation Dice"); plt.plot(df.epoch,df.val_iou,label="Validation IoU"); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0,1); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_DIR/"unet_metric_curves.png",dpi=180); plt.close()
print("Best model saved to",MODEL_DIR / "unet_best.pt")
