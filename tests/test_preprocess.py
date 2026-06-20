from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocess import apply_clahe, load_grayscale_image, normalize_image, preprocess_for_model, resize_image


def test_load_grayscale_image_reads_valid_file(tmp_path: Path):
    image_path = tmp_path / "img.png"
    image = np.ones((80, 80), dtype=np.uint8) * 100
    cv2.imwrite(str(image_path), image)

    loaded = load_grayscale_image(image_path)

    assert loaded.shape == (80, 80)
    assert loaded.dtype == np.uint8


def test_load_grayscale_image_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_grayscale_image(tmp_path / "missing.png")


def test_resize_image_returns_requested_shape():
    image = np.ones((80, 100), dtype=np.uint8)
    resized = resize_image(image, size=(128, 128))

    assert resized.shape == (128, 128)


def test_normalize_image_range():
    image = np.array([[0, 255]], dtype=np.uint8)
    normalized = normalize_image(image)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_preprocess_for_model_shape(tmp_path: Path):
    image_path = tmp_path / "img.png"
    image = np.ones((80, 80), dtype=np.uint8) * 120
    cv2.imwrite(str(image_path), image)

    processed = preprocess_for_model(image_path)

    assert processed.shape == (1, 128, 128)


def test_apply_clahe_preserves_shape():
    image = np.ones((64, 64), dtype=np.uint8) * 90
    enhanced = apply_clahe(image)

    assert enhanced.shape == image.shape
