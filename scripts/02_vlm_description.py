import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclei_pipeline.config import DATA_DIR, JSON_DIR
from nuclei_pipeline.data import image_paths
from nuclei_pipeline.ollama_client import (
    DEFAULT_VISION_MODEL,
    NAIVE_VLM_PROMPT,
    OPTIMISED_VLM_PROMPT,
    call_vision,
    extract_json,
)

parser = argparse.ArgumentParser(description="Task 1: direct VLM description and prompt comparison")
parser.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Ollama vision model")
args = parser.parse_args()

image = image_paths(DATA_DIR, "val")[0]
results = {
    "image_id": image.stem,
    "vision_model": args.model,
    "naive_prompt": NAIVE_VLM_PROMPT,
    "optimised_prompt": OPTIMISED_VLM_PROMPT,
}

results["naive_output"] = call_vision(
    image, NAIVE_VLM_PROMPT, model=args.model, temperature=0.4
)

repeats = []
for run_number in range(1, 4):
    text = call_vision(
        image, OPTIMISED_VLM_PROMPT, model=args.model, temperature=0.35
    )
    try:
        parsed = extract_json(text)
    except Exception:
        parsed = {"parse_error": True, "raw": text}
    repeats.append({"run": run_number, "output": parsed})

results["optimised_repeated_outputs"] = repeats
out_path = JSON_DIR / "vlm_prompt_comparison.json"
out_path.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print(f"\nSaved: {out_path}")
