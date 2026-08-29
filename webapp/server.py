"""DermAI inference server.

Serves the static clinical UI and exposes a single prediction endpoint that
runs the fine-tuned EfficientNetB0 checkpoint (`best_model_ft.keras`).

Preprocessing here mirrors `dermai.data._decode` exactly — decode to RGB,
bilinear resize to 224x224, and keep pixels in the raw [0, 255] range,
because EfficientNet carries its own rescaling layer. Normalising here
would double-scale the input and silently wreck the predictions.
"""

import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

IMG_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Clinical grouping. akiec/bcc/mel are malignant or pre-malignant; the rest
# are benign. The risk score is the summed probability of the malignant group.
MALIGNANT = {"akiec", "bcc", "mel"}

CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "malignant": True,
        "note": "Sun-damage lesion that can progress to squamous cell carcinoma.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "malignant": True,
        "note": "The most common skin cancer. Locally invasive, rarely spreads.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "malignant": False,
        "note": "Includes seborrhoeic keratoses and solar lentigines.",
    },
    "df": {
        "name": "Dermatofibroma",
        "malignant": False,
        "note": "Benign fibrous nodule, often on the limbs.",
    },
    "mel": {
        "name": "Melanoma",
        "malignant": True,
        "note": "The most serious skin cancer. Early detection changes outcomes.",
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "malignant": False,
        "note": "A common mole. The majority class in HAM10000.",
    },
    "vasc": {
        "name": "Vascular Lesion",
        "malignant": False,
        "note": "Angiomas, haemorrhages and related vascular findings.",
    },
}

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODEL_PATH = Path(
    os.environ.get("DERMAI_MODEL", ROOT.parent / "models" / "best_model_ft.keras")
)

app = FastAPI(title="DermAI")
_model = None


def get_model():
    """Load the checkpoint once, on first request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found at {MODEL_PATH}.",
            )
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def preprocess(raw: bytes) -> np.ndarray:
    """Bytes -> (1, 224, 224, 3) float32 tensor in [0, 255]."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # honour phone photo orientation
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file.")

    arr = tf.convert_to_tensor(np.asarray(img), dtype=tf.float32)
    arr = tf.image.resize(arr, [IMG_SIZE, IMG_SIZE])  # bilinear, as in training
    return tf.expand_dims(arr, 0).numpy()


@app.get("/api/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists(), "model": str(MODEL_PATH)}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    batch = preprocess(raw)
    probs = get_model().predict(batch, verbose=0)[0].astype(float)

    classes = [
        {
            "code": code,
            "name": CLASS_INFO[code]["name"],
            "note": CLASS_INFO[code]["note"],
            "malignant": CLASS_INFO[code]["malignant"],
            "probability": float(p),
        }
        for code, p in zip(CLASS_NAMES, probs)
    ]
    classes.sort(key=lambda c: c["probability"], reverse=True)

    malignant_score = sum(c["probability"] for c in classes if c["malignant"])
    melanoma = next(c["probability"] for c in classes if c["code"] == "mel")

    if malignant_score >= 0.50:
        band = "high"
    elif malignant_score >= 0.20:
        band = "moderate"
    else:
        band = "low"

    return {
        "top": classes[0],
        "classes": classes,
        "malignant_score": malignant_score,
        "melanoma_probability": melanoma,
        "band": band,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
