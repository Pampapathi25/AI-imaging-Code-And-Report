import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, opening, disk
from skimage.measure import label, regionprops_table


def otsu_segment(image: np.ndarray) -> np.ndarray:
    threshold = threshold_otsu(image)
    mask = image > threshold
    mask = opening(mask, disk(1))
    mask = remove_small_objects(mask, max_size=19)
    mask = remove_small_holes(mask, max_size=19)
    return mask.astype(np.uint8)


def region_features(image: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    labelled = label(mask > 0)
    props = regionprops_table(
        labelled,
        intensity_image=image,
        properties=(
            "label", "area", "eccentricity", "solidity", "mean_intensity",
            "perimeter", "equivalent_diameter_area", "extent"
        ),
    )
    return pd.DataFrame(props)


def summarise_features(df: pd.DataFrame, image_shape=(256, 256)) -> dict:
    n = len(df)
    if n == 0:
        return {
            "n_objects": 0, "mean_area": 0.0, "median_area": 0.0,
            "mean_eccentricity": 0.0, "mean_solidity": 0.0,
            "mean_intensity": 0.0, "area_fraction": 0.0,
            "density_class": "sparse", "shape_regularity": "uncertain",
        }
    total_area = float(df["area"].sum())
    area_fraction = total_area / float(image_shape[0] * image_shape[1])
    if n < 15:
        density = "sparse"
    elif n < 40:
        density = "normal"
    elif n < 65:
        density = "dense"
    else:
        density = "very_dense"
    mean_solidity = float(df["solidity"].mean())
    mean_ecc = float(df["eccentricity"].mean())
    if mean_solidity >= 0.9 and mean_ecc < 0.7:
        regularity = "mostly_regular"
    elif mean_solidity >= 0.75:
        regularity = "mixed"
    else:
        regularity = "irregular_or_clustered"
    return {
        "n_objects": int(n),
        "mean_area": float(df["area"].mean()),
        "median_area": float(df["area"].median()),
        "mean_eccentricity": mean_ecc,
        "mean_solidity": mean_solidity,
        "mean_intensity": float(df["mean_intensity"].mean()),
        "area_fraction": area_fraction,
        "density_class": density,
        "shape_regularity": regularity,
    }


def numbers_summary_text(summary: dict) -> str:
    return (
        f"Connected objects: {summary['n_objects']}. Mean area: {summary['mean_area']:.1f} px; "
        f"median area: {summary['median_area']:.1f} px; area fraction: {summary['area_fraction']:.3f}. "
        f"Mean eccentricity: {summary['mean_eccentricity']:.3f}; mean solidity: {summary['mean_solidity']:.3f}; "
        f"mean object intensity: {summary['mean_intensity']:.3f}. "
        f"Rule-based density class: {summary['density_class']}; shape regularity: {summary['shape_regularity']}."
    )
