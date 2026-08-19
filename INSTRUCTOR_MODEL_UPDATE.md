# Instructor model update applied

The course instructor advised that if `llama3.2-vision` fails with an error such as
`unknown model architecture: mllama`, students may use an alternative vision model,
including **Qwen2.5-VL**, **Qwen3-VL**, or **ministral-3:14b**, or use
`llama3.2-vision` in Colab.

This project therefore defaults to:

- **Direct image/VLM step:** `qwen2.5vl:3b` through local Ollama.
- **Numbers-only LLM and hybrid narrative steps:** `qwen2.5:3b` through local Ollama.

The prompts and assignment logic are unchanged: the vision model is descriptive rather
than diagnostic, permits `uncertain`, and returns the required structured JSON fields.
The text model never receives the image in the numbers-first and hybrid stages.

You can override either model without editing code:

```bash
python run_all.py --epochs 20 --vision-model qwen3-vl --text-model qwen2.5:3b
```

or with environment variables:

```bash
export NUCLEI_VISION_MODEL=qwen2.5vl:3b
export NUCLEI_TEXT_MODEL=qwen2.5:3b
```
