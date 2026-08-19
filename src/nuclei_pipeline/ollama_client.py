import base64
import json
import os
from pathlib import Path

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
# Lecturer-approved alternative to llama3.2-vision. Override from the shell if needed.
DEFAULT_VISION_MODEL = os.getenv("NUCLEI_VISION_MODEL", "qwen2.5vl:3b")
DEFAULT_TEXT_MODEL = os.getenv("NUCLEI_TEXT_MODEL", "qwen2.5:3b")

NAIVE_VLM_PROMPT = "Describe this biomedical image."

OPTIMISED_VLM_PROMPT = '''You are assisting with an EDUCATIONAL biomedical image-analysis exercise.
Describe only visible image characteristics. Do NOT diagnose disease, infer a patient condition,
or claim clinical significance. If a requested field cannot be determined from the image, use
"uncertain". Return valid JSON only, with exactly these keys:
{
  "modality": "...",
  "tissue_type": "...",
  "notable_features": ["..."],
  "image_quality": "..."
}
The image is from a synthetic fluorescence-microscopy nuclei dataset. Be concise and descriptive.'''

NUMBERS_PROMPT_TEMPLATE = '''You are interpreting ONLY numerical image-analysis features for an educational exercise.
You have NOT seen the image. Do not diagnose disease and do not invent visual details not supported by the numbers.
If evidence is insufficient, use "uncertain".

NUMERICAL SUMMARY:
{summary}

Return exactly two parts:
1) A single short descriptive paragraph grounded only in the numbers.
2) A valid JSON object with exactly these keys:
{{"n_objects": <integer>, "density_class": "...", "shape_regularity": "...", "quality_flag": "..."}}
Do not add clinical claims.'''

HYBRID_PROMPT_TEMPLATE = '''You are generating an auditable educational summary from segmentation-derived numbers only.
You have NOT seen the image. Do not diagnose or infer clinical significance. Keep the supplied numeric values unchanged.
If quality cannot be established, use "uncertain".

image_id: {image_id}
n_objects: {n_objects}
mean_area: {mean_area:.2f}
area_fraction: {area_fraction:.4f}
mean_eccentricity: {mean_eccentricity:.4f}
mean_solidity: {mean_solidity:.4f}
density_class: {density_class}
shape_regularity: {shape_regularity}

Return valid JSON only with exactly these keys:
{{"image_id":"{image_id}","n_objects":{n_objects},"mean_area":{mean_area:.2f},"density_class":"{density_class}","quality_flag":"...","narrative":"one concise paragraph grounded only in the supplied values"}}'''


def _call(model: str, prompt: str, images=None, temperature=0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if images:
        payload["images"] = images
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if response.status_code >= 400:
            detail = response.text.strip()
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} while running '{model}'. "
                f"Server response: {detail or 'no error body'}"
            )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at localhost:11434. Open the Ollama app or run `ollama serve`."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"Ollama timed out while running '{model}'.") from exc

    data = response.json()
    return data.get("response", "").strip()


def call_vision(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = 0.2,
) -> str:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return _call(model, prompt, images=[encoded], temperature=temperature)


def call_text(
    prompt: str,
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.2,
) -> str:
    return _call(model, prompt, temperature=temperature)


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
