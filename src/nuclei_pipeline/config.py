from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "nuclei_dataset"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
JSON_DIR = OUTPUT_DIR / "json"
METRIC_DIR = OUTPUT_DIR / "metrics"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
IMAGE_SIZE = (256, 256)
SEED = 42

for p in [OUTPUT_DIR, FIGURE_DIR, JSON_DIR, METRIC_DIR, FEATURE_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
