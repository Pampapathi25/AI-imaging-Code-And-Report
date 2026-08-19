import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib.pyplot as plt
from nuclei_pipeline.config import DATA_DIR, FIGURE_DIR
from nuclei_pipeline.data import image_paths, load_grayscale

paths = image_paths(DATA_DIR, "train")
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, p in zip(axes.ravel(), paths[:6]):
    ax.imshow(load_grayscale(p), cmap="gray")
    ax.set_title(p.stem)
    ax.axis("off")
fig.suptitle("Sample preprocessed nuclei images (grayscale, 256×256)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "eda_samples.png", dpi=180)
plt.close(fig)

pixels = np.concatenate([load_grayscale(p).ravel() for p in paths])
plt.figure(figsize=(7, 4))
plt.hist(pixels, bins=80)
plt.xlabel("Normalised grayscale intensity")
plt.ylabel("Pixel count")
plt.title("Training-set intensity histogram")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "intensity_histogram.png", dpi=180)
plt.close()
print("Saved EDA figures to", FIGURE_DIR)
