# Assignment requirements implemented by this repository

## Assessment objective
Build and evaluate a compact local biomedical image-analysis pipeline combining:

1. a multimodal vision-language model (VLM),
2. classical image processing and quantitative region features,
3. a small U-Net segmentation network,
4. an auditable hybrid pipeline producing structured records and narratives.

## Required Task 1
- Convert images to grayscale.
- Resize/use a common 256×256 size.
- Produce short EDA with sample images and an intensity histogram.
- Send a representative image to a local multimodal VLM.
- Compare a naive prompt against an optimised descriptive-not-diagnostic prompt.
- Optimised structured JSON fields: modality, tissue_type, notable_features, image_quality.
- Explicitly allow `uncertain`.
- Demonstrate repeated VLM runs are not necessarily identical.
- Include the optimised prompt(s) in the report.

**Applied instructor update:** this repository uses `qwen2.5vl:3b` rather than `llama3.2-vision` because the instructor explicitly allowed Qwen2.5-VL/Qwen3-VL/ministral vision alternatives when the `mllama` error occurs.

## Required Task 2
- Otsu thresholding.
- Morphological cleanup.
- Connected component labelling.
- `regionprops_table` quantitative feature extraction.
- Include measures such as area, eccentricity, solidity and mean intensity.
- Convert features into a short numbers-only summary.
- The text LLM must not see the image.
- Request a paragraph and JSON containing n_objects, density_class, shape_regularity, quality_flag.
- Compare numbers-first interpretation with Task 1's direct image description.

## Required Task 3
- Train the provided/small PyTorch U-Net.
- Use a modest number of epochs.
- Evaluate on held-out validation data.
- Report mean Dice and IoU.
- Show input, ground-truth mask and prediction for at least three validation images.

## Required Task 4
Run unseen test images through:

U-Net mask → regionprops feature table → structured record → narrative.

Final record fields include image_id, n_objects, mean_area, density_class and quality_flag. Aggregate records into a pandas DataFrame and save CSV.

## Required Task 5 report evidence
Maximum four pages. The code generates evidence for:
- sample images and histograms,
- U-Net validation panels,
- loss and Dice curves,
- evaluation metrics,
- example JSON records and narratives,
- optimised prompts,
- critical comparison of methods and design trade-offs.

## Five questions that final results should support
1. Direct VLM versus numbers-first: usefulness and trustworthiness.
2. U-Net versus Otsu, with one example where each does better.
3. U-Net Dice/IoU, their interpretation and failure regions/images.
4. LLM hallucination points and why structured JSON/source-of-truth design reduces risk.
5. Clinical trust, dataset limitations and the single most important change to improve trustworthiness.

## Extra-credit components provided
- Robustness tracing with supplied corrupted images.
- BCE versus Dice versus BCE+Dice loss comparison.
