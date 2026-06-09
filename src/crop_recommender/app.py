import streamlit as st

from .config import FEATURES
from .predictor import load_artifacts, predict_crop


@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


def main() -> None:
    st.set_page_config(page_title="Crop Recommendation System", layout="centered")
    st.title("Crop Recommendation System")
    st.caption("Educational ML app that recommends a crop based on soil and climate inputs.")

    st.info(
        "This is a portfolio demonstration. Local agronomy decisions should also use field testing, "
        "regional expertise, and current environmental conditions."
    )

    values = []
    for label, min_value, max_value, default_value in FEATURES:
        values.append(
            st.number_input(
                label,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=0.1,
            )
        )

    if st.button("Predict Crop", type="primary"):
        try:
            crop = predict_crop(values, artifacts=cached_artifacts())
            if crop == "Unknown crop":
                st.warning("The model returned a label that is not mapped to a crop name.")
            else:
                st.success(f"{crop} is the recommended crop for the provided conditions.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
