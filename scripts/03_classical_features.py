import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from nuclei_pipeline.config import DATA_DIR, FEATURE_DIR, JSON_DIR, FIGURE_DIR
from nuclei_pipeline.data import image_paths, load_grayscale, load_mask
from nuclei_pipeline.classical import otsu_segment, region_features, summarise_features, numbers_summary_text
from nuclei_pipeline.metrics import binary_dice, binary_iou
from nuclei_pipeline.ollama_client import NUMBERS_PROMPT_TEMPLATE, call_text

rows=[]
example=None
for p in image_paths(DATA_DIR, "val"):
    image=load_grayscale(p)
    pred=otsu_segment(image)
    gt=load_mask(DATA_DIR / "val" / "masks" / p.name)
    features=region_features(image,pred)
    features.to_csv(FEATURE_DIR / f"{p.stem}_otsu_regionprops.csv",index=False)
    summary=summarise_features(features,image.shape)
    rows.append({"image_id":p.stem,"dice":binary_dice(pred,gt),"iou":binary_iou(pred,gt),**summary})
    if example is None: example=(p,image,gt,pred,summary)
pd.DataFrame(rows).to_csv(FEATURE_DIR / "otsu_validation_metrics.csv",index=False)

p,image,gt,pred,summary=example
fig,axes=plt.subplots(1,3,figsize=(10,3.5))
for ax,arr,title in zip(axes,[image,gt,pred],["Input","Ground truth","Otsu + cleanup"]):
    ax.imshow(arr,cmap="gray"); ax.set_title(title); ax.axis("off")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "otsu_example.png",dpi=180); plt.close(fig)

text_summary=numbers_summary_text(summary)
prompt=NUMBERS_PROMPT_TEMPLATE.format(summary=text_summary)
try:
    llm_output=call_text(prompt)
except RuntimeError as exc:
    llm_output=f"OLLAMA_NOT_RUN: {exc}"
record={"image_id":p.stem,"numbers_summary":text_summary,"prompt":prompt,"llm_output":llm_output}
(JSON_DIR / "numbers_first_example.json").write_text(json.dumps(record,indent=2))
print("Saved classical features, metrics, figure, and numbers-first example.")
