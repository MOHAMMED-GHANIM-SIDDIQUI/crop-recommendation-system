from pathlib import Path
import pickle

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
STANDARD_SCALER_PATH = APP_DIR / "standscaler.pkl"
MINMAX_SCALER_PATH = APP_DIR / "minmaxscaler.pkl"

CROP_LABELS = {
    1: "Rice",
    2: "Maize",
    3: "Jute",
    4: "Cotton",
    5: "Coconut",
    6: "Papaya",
    7: "Orange",
    8: "Apple",
    9: "Muskmelon",
    10: "Watermelon",
    11: "Grapes",
    12: "Mango",
    13: "Banana",
    14: "Pomegranate",
    15: "Lentil",
    16: "Blackgram",
    17: "Mungbean",
    18: "Mothbeans",
    19: "Pigeonpeas",
    20: "Kidneybeans",
    21: "Chickpea",
    22: "Coffee",
}

FEATURES = [
    ("Nitrogen (N)", 0.0, 200.0, 50.0),
    ("Phosphorus (P)", 0.0, 200.0, 50.0),
    ("Potassium (K)", 0.0, 250.0, 50.0),
    ("Temperature (C)", -10.0, 60.0, 25.0),
    ("Humidity (%)", 0.0, 100.0, 60.0),
    ("pH", 0.0, 14.0, 6.5),
    ("Rainfall (mm)", 0.0, 500.0, 100.0),
]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [
        path.name
        for path in [MODEL_PATH, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)}")

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with STANDARD_SCALER_PATH.open("rb") as scaler_file:
        standard_scaler = pickle.load(scaler_file)
    with MINMAX_SCALER_PATH.open("rb") as scaler_file:
        minmax_scaler = pickle.load(scaler_file)
    return model, standard_scaler, minmax_scaler


def predict_crop(feature_values: list[float]) -> str:
    model, standard_scaler, minmax_scaler = load_artifacts()
    feature_array = np.asarray(feature_values, dtype=float).reshape(1, -1)
    scaled_features = minmax_scaler.transform(feature_array)
    final_features = standard_scaler.transform(scaled_features)
    prediction = int(model.predict(final_features)[0])
    return CROP_LABELS.get(prediction, "Unknown crop")


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
            crop = predict_crop(values)
            if crop == "Unknown crop":
                st.warning("The model returned a label that is not mapped to a crop name.")
            else:
                st.success(f"{crop} is the recommended crop for the provided conditions.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
