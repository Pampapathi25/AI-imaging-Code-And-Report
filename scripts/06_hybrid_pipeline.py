import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
import torch
from nuclei_pipeline.config import DATA_DIR,MODEL_DIR,JSON_DIR,FEATURE_DIR,OUTPUT_DIR
from nuclei_pipeline.data import image_paths,load_grayscale
from nuclei_pipeline.unet import SmallUNet
from nuclei_pipeline.train_utils import pick_device
from nuclei_pipeline.classical import region_features,summarise_features
from nuclei_pipeline.ollama_client import HYBRID_PROMPT_TEMPLATE,call_text,extract_json

parser=argparse.ArgumentParser(); parser.add_argument("--skip-llm",action="store_true",help="Create deterministic records without Ollama")
args=parser.parse_args(); device=pick_device()
ckpt=torch.load(MODEL_DIR/"unet_best.pt",map_location=device); model=SmallUNet().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
records=[]
for p in image_paths(DATA_DIR,"test"):
    image=load_grayscale(p); x=torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad(): mask=(torch.sigmoid(model(x))[0,0].cpu().numpy()>=0.5).astype("uint8")
    features=region_features(image,mask); features.to_csv(FEATURE_DIR/f"{p.stem}_unet_regionprops.csv",index=False)
    s=summarise_features(features,image.shape)
    base={"image_id":p.stem,"n_objects":s["n_objects"],"mean_area":round(s["mean_area"],2),"density_class":s["density_class"],"quality_flag":"uncertain"}
    prompt=HYBRID_PROMPT_TEMPLATE.format(image_id=p.stem,**s)
    if args.skip_llm:
        record={**base,"narrative":f"Segmentation-derived analysis identified {s['n_objects']} objects with mean area {s['mean_area']:.1f} pixels and {s['density_class']} density. Shape regularity was {s['shape_regularity']}. Quality remains uncertain without independent quality assessment."}
    else:
        try:
            record=extract_json(call_text(prompt))
            for k in ["image_id","n_objects","mean_area","density_class"]: record[k]=base[k]
        except Exception as exc:
            record={**base,"narrative":f"LLM unavailable or invalid output: {exc}"}
    records.append(record); (JSON_DIR/f"{p.stem}.json").write_text(json.dumps(record,indent=2))
out=pd.DataFrame(records); out.to_csv(OUTPUT_DIR/"hybrid_test_records.csv",index=False)
print(out.to_string(index=False)); print("Saved",OUTPUT_DIR/"hybrid_test_records.csv")
