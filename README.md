# Hybrid Biomedical Nuclei Image Analysis — Assignment 3

A complete VS Code-ready implementation of the supplied biomedical image-analysis assignment using the provided **synthetic fluorescence-microscopy nuclei dataset**.

The project implements this auditable pipeline:

**raw image → preprocessing/EDA → direct VLM description → Otsu segmentation + region features → numbers-first LLM description → U-Net segmentation → region features → structured JSON → narrative → aggregated CSV**

> Educational use only. The system is not clinically validated and must not be used for diagnosis or patient care.

## Important model change from the instructor

The original brief names `llama3.2-vision`. The instructor later advised that students who encounter the `unknown model architecture: mllama` error may use an alternative such as **Qwen2.5-VL, Qwen3-VL, or ministral-3:14b**, or run Llama 3.2 Vision in Colab.

This repository therefore uses:

- **Vision model:** `qwen2.5vl:3b` (default)
- **Text-only model:** `qwen2.5:3b` (default)

Both run locally through Ollama. The model names are configurable, so the rest of the code does not need to change if you later choose Qwen3-VL or another instructor-approved model.

See `INSTRUCTOR_MODEL_UPDATE.md` for the applied change.

## Assignment coverage

### Task 1 — Data preparation and multimodal LLM description
- Uses grayscale 256×256 images from the supplied dataset.
- Produces an EDA sample panel and intensity histogram.
- Sends a representative validation image to local `qwen2.5vl:3b` through Ollama.
- Compares a naive prompt with an optimised descriptive-not-diagnostic prompt.
- The optimised prompt forces JSON fields: `modality`, `tissue_type`, `notable_features`, `image_quality`.
- Explicitly permits `uncertain`.
- Repeats the optimised prompt three times to show run-to-run variability.
- Saves the prompts and outputs for direct use as report evidence.

### Task 2 — Classical features and numbers-first LLM interpretation
- Applies Otsu thresholding and morphological cleanup.
- Labels connected components.
- Uses `regionprops_table` to compute object features including area, eccentricity, solidity and mean intensity.
- Saves per-image feature tables.
- Converts the features into a numerical natural-language summary.
- Sends **numbers only** to local `qwen2.5:3b`; the text model never sees the image.
- Requests a paragraph plus the structured fields `n_objects`, `density_class`, `shape_regularity`, `quality_flag`.

### Task 3 — U-Net segmentation
- Trains a compact PyTorch U-Net on the supplied training split.
- Uses the supplied validation split for evaluation.
- Saves Dice and IoU metrics.
- Generates training loss and validation metric curves.
- Generates side-by-side input / ground-truth / U-Net prediction panels for at least three validation images.

### Task 4 — Hybrid pipeline
For every unseen test image:

**U-Net mask → regionprops table → numerical summary → LLM structured JSON → one-paragraph narrative**

The final JSON keeps the deterministic segmentation-derived values as the source of truth and aggregates all test records into:

`outputs/hybrid_test_records.csv`

### Task 5 — Report evidence
The code generates the figures, prompts, structured outputs and numerical evidence required for the maximum four-page report. Use your actual generated results rather than inventing values.

### Extra-credit work included
- `07_loss_ablation.py`: compares BCE, Dice and BCE+Dice.
- `08_robustness.py`: traces the supplied corrupted test images through the U-Net and quantitative feature stage.

## Project structure

```text
biomedical_nuclei_assignment/
├── data/
│   └── nuclei_dataset/            # supplied train/val/test/corrupted data
├── src/
│   └── nuclei_pipeline/
│       ├── config.py
│       ├── data.py
│       ├── classical.py
│       ├── metrics.py
│       ├── ollama_client.py
│       ├── train_utils.py
│       └── unet.py
├── scripts/
│   ├── 01_eda.py
│   ├── 02_vlm_description.py
│   ├── 03_classical_features.py
│   ├── 04_train_unet.py
│   ├── 05_evaluate_unet.py
│   ├── 06_hybrid_pipeline.py
│   ├── 07_loss_ablation.py
│   └── 08_robustness.py
├── outputs/                        # created automatically when scripts run
├── ASSIGNMENT_REQUIREMENTS.md
├── INSTRUCTOR_MODEL_UPDATE.md
├── requirements.txt
├── run_all.py
└── README.md
```

# Run in VS Code on macOS

## 1. Extract and open the project

Extract the ZIP, then in VS Code choose:

**File → Open Folder → biomedical_nuclei_assignment**

Open the VS Code terminal with:

**Terminal → New Terminal**

## 2. Create the Python virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If possible, Python 3.11 or 3.12 is recommended for the smoothest PyTorch experience. If your existing Python 3.14 environment installs all requirements successfully, you may continue with it.

In VS Code select the environment:

`Cmd + Shift + P` → **Python: Select Interpreter** → choose `.venv/bin/python`.

## 3. Make sure Ollama is installed and running

Open the Ollama application on your Mac.

Check it:

```bash
ollama --version
```

The old `llama3.2-vision` model is **not required** for this version.

## 4. Pull the lecturer-approved Qwen models

Run:

```bash
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:3b
```

Check that they are installed:

```bash
ollama list
```

You should see both model names.

## 5. Test the vision model before running Python

Run:

```bash
ollama run qwen2.5vl:3b
```

If the `>>>` prompt appears, type:

```text
Reply with OK.
```

Exit with:

```text
/bye
```

You can also test it with one supplied image from the project root:

```bash
ollama run qwen2.5vl:3b data/nuclei_dataset/val/images/val_000.png "Describe only visible image characteristics."
```

If that exact filename does not exist, list one first:

```bash
ls data/nuclei_dataset/val/images | head
```

Then substitute the displayed filename.

## 6. Test the text-only model

```bash
ollama run qwen2.5:3b "Reply only with OK"
```

## 7. First run the non-LLM computer-vision pipeline

This checks the dataset, Otsu code, U-Net training and evaluation before involving Ollama:

```bash
python run_all.py --epochs 5 --skip-llm
```

This is only a test run. Do not use five-epoch smoke-test metrics as your final report results.

## 8. Run Task 1 by itself

```bash
python scripts/02_vlm_description.py
```

It will use `qwen2.5vl:3b` automatically and save:

```text
outputs/json/vlm_prompt_comparison.json
```

## 9. Run the complete assignment

For the final experiment:

```bash
python run_all.py --epochs 20
```

The default models are automatically passed to all LLM stages.

You may explicitly state them if desired:

```bash
python run_all.py --epochs 20 --vision-model qwen2.5vl:3b --text-model qwen2.5:3b
```

## Run each required task individually

```bash
python scripts/01_eda.py
python scripts/02_vlm_description.py
python scripts/03_classical_features.py
python scripts/04_train_unet.py --epochs 20 --loss bce+dice
python scripts/05_evaluate_unet.py
python scripts/06_hybrid_pipeline.py
```

## Optional extra-credit experiments

Loss ablation:

```bash
python scripts/07_loss_ablation.py --epochs 10
```

Robustness analysis after a trained U-Net exists:

```bash
python scripts/08_robustness.py
```

# Changing to another instructor-approved vision model

The vision model can be changed without editing Python code.

For example, after pulling a compatible Qwen3-VL model in Ollama:

```bash
python run_all.py --epochs 20 --vision-model qwen3-vl
```

Or set environment variables:

```bash
export NUCLEI_VISION_MODEL=qwen2.5vl:3b
export NUCLEI_TEXT_MODEL=qwen2.5:3b
python run_all.py --epochs 20
```

# Main final outputs

After a complete run, inspect:

```text
outputs/
├── figures/
│   ├── eda_samples.png
│   ├── intensity_histogram.png
│   ├── otsu_example.png
│   ├── unet_loss_curve.png
│   ├── unet_metric_curves.png
│   └── unet_validation_examples.png
├── metrics/
│   ├── unet_history.csv
│   ├── validation_comparison.csv
│   └── evaluation_summary.csv
├── features/
│   └── *_regionprops.csv
├── json/
│   ├── vlm_prompt_comparison.json
│   ├── numbers_first_example.json
│   └── test-image JSON records
├── models/
│   └── unet_best.pt
└── hybrid_test_records.csv
```

# Evidence to use in the four-page report

Use these outputs to answer the five required questions:

1. Compare the direct VLM output in `vlm_prompt_comparison.json` with the numbers-first output in `numbers_first_example.json` for usefulness versus trustworthiness.
2. Use `validation_comparison.csv` to compare U-Net and Otsu and identify one image where each method performs better.
3. Report the actual mean U-Net Dice and IoU from `evaluation_summary.csv`, then use per-image metrics and the validation panels to discuss failure cases.
4. Explain that hallucination can occur at the direct VLM, numbers-first interpretation and final narrative stages. The segmentation masks, feature tables and structured deterministic fields provide an auditable source of truth.
5. Explain that this is a small synthetic educational dataset and not a clinically validated system; discuss what additional validation/data change would most improve trustworthiness.

Also include the exact optimised prompts. They are stored in:

`src/nuclei_pipeline/ollama_client.py`

and copied into the generated JSON evidence.

# Troubleshooting

## `unknown model architecture: mllama`

That is the issue that affected `llama3.2-vision`. Do **not** use that model in this revised project. Run:

```bash
ollama pull qwen2.5vl:3b
ollama run qwen2.5vl:3b
```

Then run the assignment again.

## Cannot connect to Ollama

Open the Ollama application or run:

```bash
ollama serve
```

Then test:

```bash
curl http://localhost:11434/api/tags
```

## Ollama says a model is missing

```bash
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:3b
```

## U-Net training is slow

First verify the project with:

```bash
python run_all.py --epochs 5 --skip-llm
```

Then use 20 epochs for the final experiment. Apple Silicon PyTorch uses MPS when available; otherwise the code falls back to CPU.

## A script fails after you have changed folders

Always return to the project root before running scripts:

```bash
cd ~/Downloads/biomedical_nuclei_assignment
source .venv/bin/activate
```

# Reproducibility and marking

The code is separated into reusable functions/modules, has a single top-level runner, saves numerical values and figures used in the report, preserves the exact prompts, and writes auditable per-image JSON/CSV records. This is intended to make the analysis re-runnable by a marker rather than relying on manually copied or invented outputs.
