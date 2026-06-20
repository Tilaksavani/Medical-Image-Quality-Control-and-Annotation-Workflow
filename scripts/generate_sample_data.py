from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.qc import generate_quality_report


IMAGE_DIR = ROOT / "data" / "sample_images"
ARTIFACT_DIR = ROOT / "artifacts"


def draw_normal_image(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8) + 35
    center = (int(rng.integers(45, 83)), int(rng.integers(45, 83)))
    axes = (int(rng.integers(22, 38)), int(rng.integers(28, 45)))
    angle = float(rng.integers(0, 180))
    cv2.ellipse(image, center, axes, angle, 0, 360, int(rng.integers(150, 220)), -1)
    noise = rng.normal(0, 8, size=(size, size))
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def draw_low_contrast_image(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8) + 95
    center = (int(rng.integers(45, 83)), int(rng.integers(45, 83)))
    radius = int(rng.integers(22, 40))
    cv2.circle(image, center, radius, int(rng.integers(105, 125)), -1)
    noise = rng.normal(0, 3, size=(size, size))
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def draw_artifact_image(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    image = draw_normal_image(rng, size=size)
    for _ in range(3):
        x1 = int(rng.integers(0, size))
        y1 = int(rng.integers(0, size))
        x2 = int(rng.integers(0, size))
        y2 = int(rng.integers(0, size))
        cv2.line(image, (x1, y1), (x2, y2), 255, int(rng.integers(1, 3)))
    if rng.random() > 0.5:
        image = cv2.GaussianBlur(image, (7, 7), 0)
    return image


def main() -> None:
    rng = np.random.default_rng(42)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in IMAGE_DIR.glob("*.png"):
        old_file.unlink()

    rows = []
    labels = ["normal", "low_contrast", "artifact"]
    generator_map = {
        "normal": draw_normal_image,
        "low_contrast": draw_low_contrast_image,
        "artifact": draw_artifact_image,
    }

    image_id = 0
    for label in labels:
        for _ in range(40):
            image_id += 1
            file_name = f"sample_{image_id:03d}_{label}.png"
            image = generator_map[label](rng)
            image_path = IMAGE_DIR / file_name
            cv2.imwrite(str(image_path), image)
            report = generate_quality_report(image_path).to_dict()
            rows.append({
                "file_name": file_name,
                "label": label,
                "qc_overall_pass": report["overall_pass"],
                "blur_score": report["blur_score"],
                "contrast_score": report["contrast_score"],
                "brightness_score": report["brightness_score"],
            })

    annotations = pd.DataFrame(rows)
    annotations.to_csv(ARTIFACT_DIR / "annotations.csv", index=False)
    print(f"Generated {len(annotations)} images in {IMAGE_DIR}")
    print(f"Wrote annotations to {ARTIFACT_DIR / 'annotations.csv'}")


if __name__ == "__main__":
    main()
