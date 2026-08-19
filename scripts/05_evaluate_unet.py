import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib.pyplot as plt
import torch
from nuclei_pipeline.config import DATA_DIR, MODEL_DIR, METRIC_DIR, FIGURE_DIR
from nuclei_pipeline.data import NucleiDataset
from nuclei_pipeline.unet import SmallUNet
from nuclei_pipeline.train_utils import pick_device
from nuclei_pipeline.metrics import binary_dice,binary_iou
from nuclei_pipeline.classical import otsu_segment

device=pick_device(); ds=NucleiDataset(DATA_DIR,"val")
ckpt=torch.load(MODEL_DIR/"unet_best.pt",map_location=device)
model=SmallUNet().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
rows=[]; panels=[]
for i in range(len(ds)):
    x,y,image_id=ds[i]
    with torch.no_grad(): pred=(torch.sigmoid(model(x.unsqueeze(0).to(device)))[0,0].cpu().numpy()>=0.5).astype("uint8")
    image=x[0].numpy(); gt=y[0].numpy().astype("uint8"); otsu=otsu_segment(image)
    rows.append({"image_id":image_id,"unet_dice":binary_dice(pred,gt),"unet_iou":binary_iou(pred,gt),"otsu_dice":binary_dice(otsu,gt),"otsu_iou":binary_iou(otsu,gt)})
    if len(panels)<3: panels.append((image_id,image,gt,pred))
df=pd.DataFrame(rows); df.to_csv(METRIC_DIR/"validation_comparison.csv",index=False)
summary=pd.DataFrame([{"method":"U-Net","mean_dice":df.unet_dice.mean(),"mean_iou":df.unet_iou.mean()},{"method":"Otsu","mean_dice":df.otsu_dice.mean(),"mean_iou":df.otsu_iou.mean()}])
summary.to_csv(METRIC_DIR/"evaluation_summary.csv",index=False)
fig,axes=plt.subplots(3,3,figsize=(9,9))
for r,(name,image,gt,pred) in enumerate(panels):
    for c,(arr,title) in enumerate([(image,f"{name}: input"),(gt,"Ground truth"),(pred,"U-Net prediction")]):
        axes[r,c].imshow(arr,cmap="gray"); axes[r,c].set_title(title); axes[r,c].axis("off")
fig.tight_layout(); fig.savefig(FIGURE_DIR/"unet_validation_examples.png",dpi=180); plt.close(fig)
print(summary.to_string(index=False))
