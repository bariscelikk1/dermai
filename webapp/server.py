"""DermAI inference server.

Serves the static clinical UI and exposes a single prediction endpoint that
runs the fine-tuned EfficientNetB0 checkpoint, converted to TensorFlow Lite
(`best_model_ft.tflite`).

TensorFlow itself is not imported here. The full package installs to ~1.4 GB,
which blew past Vercel's 500 MB function limit; the Lite runtime is a few MB
because it only runs models, it cannot train them. The conversion cost is a
shift of ~0.01 in the reported probabilities, which never changed the top
class in testing.

Preprocessing here mirrors `dermai.data._decode` exactly — decode to RGB,
bilinear resize to 224x224, and keep pixels in the raw [0, 255] range,
because EfficientNet carries its own rescaling layer. Normalising here
would double-scale the input and silently wreck the predictions.
"""

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

IMG_SIZE = 224
# Starlette buffers an upload in memory only up to 1 MB and spools anything
# larger to a temporary file. The analyse page promises the image is never
# written to disk, so refuse oversized uploads rather than break that promise.
# The browser downscales to a 1024px JPEG (~300 KB); 1 MB is ample headroom.
MAX_UPLOAD_BYTES = 1024 * 1024
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

ROOT = Path(__file__).resolve().parent.parent
# Static pages live in public/ so Vercel's CDN serves them directly; only
# /api/* reaches this process. Locally, uvicorn serves them from the same dir.
STATIC = ROOT / "public"
MODEL_PATH = Path(
    os.environ.get("DERMAI_MODEL", ROOT / "models" / "best_model_ft.tflite")
)

app = FastAPI(title="DermAI")
_model = None


def _interpreter_class():
    """LiteRT in deployment; fall back to TensorFlow's copy for local dev."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # a dev machine with full TensorFlow installed
        import tensorflow as tf

        return tf.lite.Interpreter
    return Interpreter


def get_model():
    """Load the checkpoint once, on first request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found at {MODEL_PATH}.",
            )
        interpreter = _interpreter_class()(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        _model = interpreter
    return _model


def resize_bilinear(arr: np.ndarray, size: int) -> np.ndarray:
    """Half-pixel-centre bilinear resize, no antialiasing.

    Reproduces `tf.image.resize` to within floating-point rounding. Pillow's
    own resize is not equivalent: it antialiases when shrinking, so it would
    feed the model smoother images than training ever saw.
    """
    src_h, src_w = arr.shape[:2]

    def axis(n_out: int, n_in: int):
        pos = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        pos = np.clip(pos, 0, n_in - 1)
        low = np.floor(pos).astype(int)
        high = np.minimum(low + 1, n_in - 1)
        return low, high, (pos - low).astype(np.float32)

    y0, y1, wy = axis(size, src_h)
    x0, x1, wx = axis(size, src_w)
    top_left, top_right = arr[y0][:, x0], arr[y0][:, x1]
    bot_left, bot_right = arr[y1][:, x0], arr[y1][:, x1]
    top = top_left + (top_right - top_left) * wx[None, :, None]
    bottom = bot_left + (bot_right - bot_left) * wx[None, :, None]
    return top + (bottom - top) * wy[:, None, None]


def preprocess(raw: bytes) -> np.ndarray:
    """Bytes -> (1, 224, 224, 3) float32 tensor in [0, 255]."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # honour phone photo orientation
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file.")

    arr = np.asarray(img, dtype=np.float32)
    arr = resize_bilinear(arr, IMG_SIZE)  # bilinear, as in training
    return arr[None].astype(np.float32)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists(), "model": str(MODEL_PATH)}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is larger than 1 MB after downscaling. "
            "Please submit a smaller image.",
        )

    batch = preprocess(raw)
    interpreter = get_model()
    inputs = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()[0]
    interpreter.set_tensor(inputs["index"], batch)
    interpreter.invoke()
    probs = interpreter.get_tensor(outputs["index"])[0].astype(float)

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


# Locally this process serves the site as well as the API, so uvicorn alone
# is enough to run the whole thing. On Vercel the pages come off the CDN and
# public/ is never copied into the function — and StaticFiles raises on
# construction when its directory is missing, which crashed the function at
# import time before it could answer anything. Mount only if the directory
# is actually there.
if STATIC.is_dir():

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/", StaticFiles(directory=STATIC), name="static")
