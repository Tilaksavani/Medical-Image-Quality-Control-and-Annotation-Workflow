from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 128)


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load an image as grayscale.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if OpenCV cannot decode the image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize image to a fixed size for model input."""
    if image.ndim != 2:
        raise ValueError("Expected a grayscale 2D image.")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale image values to the [0, 1] range."""
    if image.size == 0:
        raise ValueError("Cannot normalize an empty image.")
    return image.astype(np.float32) / 255.0


def preprocess_for_model(image_path: str | Path, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load, resize, normalize, and add channel dimension for CNN input."""
    image = load_grayscale_image(image_path)
    image = resize_image(image, size=size)
    image = normalize_image(image)
    return image[None, :, :]  # shape: (1, H, W)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE contrast enhancement to grayscale image."""
    if image.ndim != 2:
        raise ValueError("Expected a grayscale 2D image.")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image.astype(np.uint8))
