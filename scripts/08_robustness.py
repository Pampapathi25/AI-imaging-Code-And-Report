"""Optional extra credit: compares original versus supplied corrupted test images."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd, torch
from nuclei_pipeline.config import DATA_DIR,MODEL_DIR,OUTPUT_DIR
from nuclei_pipeline.data import load_grayscale
from nuclei_pipeline.unet import SmallUNet
from nuclei_pipeline.train_utils import pick_device
from nuclei_pipeline.classical import region_features,summarise_features

device=pick_device(); ckpt=torch.load(MODEL_DIR/"unet_best.pt",map_location=device); model=SmallUNet().to(device); model.load_state_dict(ckpt["model_state"]); model.eval(); rows=[]
for cp in sorted((DATA_DIR/"test_corrupted"/"images").glob("*.png")):
    candidates=[DATA_DIR/"test"/"images"/cp.name]
    # supplied corrupted filenames may include a suffix; match by shared test id if needed
    stem=cp.stem.split("_")[0]+"_"+cp.stem.split("_")[1] if cp.stem.startswith("test_") else cp.stem
    matches=list((DATA_DIR/"test"/"images").glob(stem+"*.png")); op=matches[0] if matches else candidates[0]
    if not op.exists(): continue
    for tag,p in [("original",op),("corrupted",cp)]:
        im=load_grayscale(p); x=torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): mask=(torch.sigmoid(model(x))[0,0].cpu().numpy()>=0.5).astype("uint8")
        s=summarise_features(region_features(im,mask),im.shape); rows.append({"pair":cp.stem,"version":tag,**s})
pd.DataFrame(rows).to_csv(OUTPUT_DIR/"robustness_feature_trace.csv",index=False); print("Saved robustness_feature_trace.csv")
