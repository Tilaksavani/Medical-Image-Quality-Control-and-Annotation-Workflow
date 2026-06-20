from pathlib import Path

import cv2
import numpy as np

from src.qc import aspect_ratio_pass, batch_quality_reports, blur_score, contrast_score, generate_quality_report, resolution_pass


def test_resolution_pass_true_for_large_image():
    assert resolution_pass(128, 128) is True


def test_aspect_ratio_pass_rejects_extreme_ratio():
    assert aspect_ratio_pass(300, 50) is False


def test_blur_and_contrast_scores_are_numeric():
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(image, (32, 32), 15, 200, -1)

    assert isinstance(blur_score(image), float)
    assert isinstance(contrast_score(image), float)


def test_generate_quality_report_valid_image(tmp_path: Path):
    image_path = tmp_path / "valid.png"
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(image, (64, 64), 35, 200, -1)
    cv2.imwrite(str(image_path), image)

    report = generate_quality_report(image_path)

    assert report.readable is True
    assert report.width == 128
    assert report.height == 128


def test_generate_quality_report_corrupted_file(tmp_path: Path):
    image_path = tmp_path / "bad.png"
    image_path.write_text("not an image")

    report = generate_quality_report(image_path)

    assert report.readable is False
    assert report.overall_pass is False


def test_batch_quality_reports_counts_images(tmp_path: Path):
    for i in range(2):
        image_path = tmp_path / f"img_{i}.png"
        image = np.ones((128, 128), dtype=np.uint8) * 120
        cv2.imwrite(str(image_path), image)

    reports = batch_quality_reports(tmp_path)

    assert len(reports) == 2
