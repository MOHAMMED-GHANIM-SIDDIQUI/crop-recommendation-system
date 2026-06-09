from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
STANDARD_SCALER_PATH = MODEL_DIR / "standscaler.pkl"
MINMAX_SCALER_PATH = MODEL_DIR / "minmaxscaler.pkl"

FEATURES = [
    ("Nitrogen (N)", 0.0, 200.0, 50.0),
    ("Phosphorus (P)", 0.0, 200.0, 50.0),
    ("Potassium (K)", 0.0, 250.0, 50.0),
    ("Temperature (C)", -10.0, 60.0, 25.0),
    ("Humidity (%)", 0.0, 100.0, 60.0),
    ("pH", 0.0, 14.0, 6.5),
    ("Rainfall (mm)", 0.0, 500.0, 100.0),
]

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
