import pickle

import numpy as np

from .config import CROP_LABELS, MINMAX_SCALER_PATH, MODEL_PATH, STANDARD_SCALER_PATH


def validate_feature_values(feature_values: list[float]) -> None:
    if len(feature_values) != 7:
        raise ValueError("Expected exactly 7 crop input features.")
    for value in feature_values:
        if not isinstance(value, (int, float)):
            raise TypeError("All crop input features must be numeric.")


def crop_name_from_label(label: int) -> str:
    return CROP_LABELS.get(int(label), "Unknown crop")


def load_pickle_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path.name}")
    with path.open("rb") as artifact_file:
        return pickle.load(artifact_file)


def load_artifacts():
    return (
        load_pickle_artifact(MODEL_PATH),
        load_pickle_artifact(STANDARD_SCALER_PATH),
        load_pickle_artifact(MINMAX_SCALER_PATH),
    )


def predict_crop(feature_values: list[float], artifacts=None) -> str:
    validate_feature_values(feature_values)
    model, standard_scaler, minmax_scaler = artifacts or load_artifacts()
    feature_array = np.asarray(feature_values, dtype=float).reshape(1, -1)
    scaled_features = minmax_scaler.transform(feature_array)
    final_features = standard_scaler.transform(scaled_features)
    prediction = model.predict(final_features)[0]
    return crop_name_from_label(int(prediction))
