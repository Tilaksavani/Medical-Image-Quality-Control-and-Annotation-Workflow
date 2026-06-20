from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.preprocess import load_grayscale_image


@dataclass
class QualityReport:
    file_name: str
    readable: bool
    width: int
    height: int
    aspect_ratio: float
    blur_score: float
    contrast_score: float
    brightness_score: float
    resolution_pass: bool
    aspect_ratio_pass: bool
    blur_pass: bool
    contrast_pass: bool
    brightness_pass: bool
    overall_pass: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def blur_score(image: np.ndarray) -> float:
    """Variance of Laplacian. Lower value usually means blurrier image."""
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def contrast_score(image: np.ndarray) -> float:
    """Standard deviation of pixel intensities."""
    return float(np.std(image))


def brightness_score(image: np.ndarray) -> float:
    """Average pixel intensity."""
    return float(np.mean(image))


def resolution_pass(width: int, height: int, min_width: int = 64, min_height: int = 64) -> bool:
    return width >= min_width and height >= min_height


def aspect_ratio_pass(width: int, height: int, min_ratio: float = 0.75, max_ratio: float = 1.33) -> bool:
    if height == 0:
        return False
    ratio = width / height
    return min_ratio <= ratio <= max_ratio


def generate_quality_report(
    image_path: str | Path,
    min_width: int = 64,
    min_height: int = 64,
    min_blur: float = 15.0,
    min_contrast: float = 8.0,
    min_brightness: float = 20.0,
    max_brightness: float = 235.0,
) -> QualityReport:
    """Run automated QC checks on one image."""
    image_path = Path(image_path)

    try:
        image = load_grayscale_image(image_path)
        readable = True
        height, width = image.shape[:2]
        ratio = width / height if height else 0.0
        blur = blur_score(image)
        contrast = contrast_score(image)
        brightness = brightness_score(image)
    except Exception:
        return QualityReport(
            file_name=image_path.name,
            readable=False,
            width=0,
            height=0,
            aspect_ratio=0.0,
            blur_score=0.0,
            contrast_score=0.0,
            brightness_score=0.0,
            resolution_pass=False,
            aspect_ratio_pass=False,
            blur_pass=False,
            contrast_pass=False,
            brightness_pass=False,
            overall_pass=False,
        )

    res_ok = resolution_pass(width, height, min_width, min_height)
    ratio_ok = aspect_ratio_pass(width, height)
    blur_ok = blur >= min_blur
    contrast_ok = contrast >= min_contrast
    brightness_ok = min_brightness <= brightness <= max_brightness
    overall = all([readable, res_ok, ratio_ok, blur_ok, contrast_ok, brightness_ok])

    return QualityReport(
        file_name=image_path.name,
        readable=readable,
        width=width,
        height=height,
        aspect_ratio=round(ratio, 3),
        blur_score=round(blur, 3),
        contrast_score=round(contrast, 3),
        brightness_score=round(brightness, 3),
        resolution_pass=res_ok,
        aspect_ratio_pass=ratio_ok,
        blur_pass=blur_ok,
        contrast_pass=contrast_ok,
        brightness_pass=brightness_ok,
        overall_pass=overall,
    )


def batch_quality_reports(image_dir: str | Path) -> list[dict[str, object]]:
    """Run QC checks over all PNG/JPG/JPEG files in a folder."""
    image_dir = Path(image_dir)
    image_paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    )
    return [generate_quality_report(path).to_dict() for path in image_paths]
