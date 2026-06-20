from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataset import ID_TO_LABEL
from src.model import SimpleCNN
from src.preprocess import apply_clahe, load_grayscale_image, preprocess_for_model
from src.qc import generate_quality_report


IMAGE_DIR = ROOT / "data" / "sample_images"
ANNOTATION_PATH = ROOT / "artifacts" / "annotations.csv"
MODEL_PATH = ROOT / "models" / "simple_cnn.pt"


@st.cache_data
def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["file_name", "label"])
    return pd.read_csv(path)


@st.cache_resource
def load_model() -> SimpleCNN | None:
    if not MODEL_PATH.exists():
        return None
    model = SimpleCNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_label(model: SimpleCNN, image_path: Path) -> str:
    image = preprocess_for_model(image_path)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_id = int(model(tensor).argmax(dim=1).item())
    return ID_TO_LABEL[pred_id]


def main() -> None:
    st.set_page_config(page_title="Medical Image QC Workflow", layout="wide")
    st.title("Medical Image Quality Control and Annotation Workflow")
    st.caption("Prototype for preprocessing, QC validation, annotation review, and CNN prediction preview.")

    annotations = load_annotations(ANNOTATION_PATH)
    if annotations.empty:
        st.warning("No annotations found. Run: python scripts/generate_sample_data.py")
        return

    model = load_model()
    image_names = annotations["file_name"].tolist()
    selected = st.sidebar.selectbox("Select image", image_names)
    selected_path = IMAGE_DIR / selected
    row = annotations[annotations["file_name"] == selected].iloc[0]

    report = generate_quality_report(selected_path).to_dict()
    image = load_grayscale_image(selected_path)
    enhanced = apply_clahe(image)

    left, middle, right = st.columns([1, 1, 1])

    with left:
        st.subheader("Original Image")
        st.image(image, clamp=True, channels="GRAY", use_container_width=True)
        st.write(f"Current label: `{row['label']}`")

    with middle:
        st.subheader("CLAHE Preview")
        st.image(enhanced, clamp=True, channels="GRAY", use_container_width=True)
        new_label = st.selectbox("Update label", ["normal", "low_contrast", "artifact"], index=["normal", "low_contrast", "artifact"].index(row["label"]))
        if st.button("Save label update"):
            annotations.loc[annotations["file_name"] == selected, "label"] = new_label
            annotations.to_csv(ANNOTATION_PATH, index=False)
            st.success("Annotation updated and saved to CSV.")

    with right:
        st.subheader("QC Report")
        st.json(report)
        if model is None:
            st.info("Train a model to preview predictions: python -m src.train --epochs 5")
        else:
            st.metric("CNN prediction", predict_label(model, selected_path))

    st.subheader("Annotation table")
    st.dataframe(annotations, use_container_width=True)
    st.download_button(
        label="Download annotations CSV",
        data=annotations.to_csv(index=False),
        file_name="annotations_updated.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
