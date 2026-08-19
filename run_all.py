import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Run the biomedical nuclei assignment pipeline")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument(
    "--skip-llm",
    action="store_true",
    help="Skip Ollama calls; useful for checking the computer-vision pipeline first",
)
parser.add_argument(
    "--vision-model",
    default=os.getenv("NUCLEI_VISION_MODEL", "qwen2.5vl:3b"),
    help="Ollama VLM for Task 1 (default: qwen2.5vl:3b)",
)
parser.add_argument(
    "--text-model",
    default=os.getenv("NUCLEI_TEXT_MODEL", "qwen2.5:3b"),
    help="Ollama text model for numbers-first/hybrid descriptions (default: qwen2.5:3b)",
)
args = parser.parse_args()


def run(script, *extra, env=None):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, extra)]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


env = os.environ.copy()
env["NUCLEI_VISION_MODEL"] = args.vision_model
env["NUCLEI_TEXT_MODEL"] = args.text_model

run("01_eda.py", env=env)
if not args.skip_llm:
    run("02_vlm_description.py", "--model", args.vision_model, env=env)
run("03_classical_features.py", env=env)
run("04_train_unet.py", "--epochs", args.epochs, env=env)
run("05_evaluate_unet.py", env=env)
run("06_hybrid_pipeline.py", *(["--skip-llm"] if args.skip_llm else []), env=env)
print("\nPipeline complete. See outputs/ for figures, metrics, JSON and CSV files.")
