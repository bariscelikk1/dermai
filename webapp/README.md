# DermAI Web Interface

A clinical-style web front end for the trained DermAI classifier. Upload a
dermoscopic image, watch the preprocessing pipeline run, and get a risk-banded
assessment with the full 7-class probability distribution.

## What it does

1. **Upload**: drag-and-drop or file picker, processed in memory (never written to disk).
2. **Analysis**: the six preprocessing/inference stages are shown as they run.
3. **Assessment**: a green / amber / red risk band derived from the summed
   probability of the malignant group (`mel`, `bcc`, `akiec`), plus melanoma
   probability called out separately and a full probability table.
4. **Referral**: on amber or red, optional browser geolocation builds Google
   Maps and Apple Maps search links for nearby dermatologists and hospitals,
   alongside curated guidance links. Coordinates stay in the browser and are
   never sent to the server.

## Layout

```
webapp/
  server.py            FastAPI app: /api/health, /api/predict, static serving
  static/
    index.html         page structure
    styles.css         design system
    app.js             upload → scan → render flow
models/
  best_model_ft.keras  the fine-tuned checkpoint (gitignored, ~49 MB)
```

## Running it

The model checkpoint must be at `models/best_model_ft.keras` (override with the
`DERMAI_MODEL` environment variable).

```bash
cd ~/Desktop/dermai && .venv/bin/python -m uvicorn webapp.server:app --port 8000
```

Then open <http://127.0.0.1:8000>.

To recreate the environment from scratch:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r webapp/requirements.txt
```

## Preprocessing contract

`server.preprocess` deliberately mirrors `dermai.data._decode`:

- decode to RGB (with EXIF orientation applied, since phone photos carry it)
- **bilinear** resize to 224 × 224
- **keep pixels in raw `[0, 255]`**

That last point is the one that breaks silently if changed. EfficientNet carries
its own rescaling layer internally, so normalising to `[0, 1]` here would
double-scale the input. The model still returns confident-looking probabilities,
they are just wrong.

## Limitations worth stating out loud

- Not a medical device, and not a diagnosis.
- Trained only on **dermoscopic** images (contact dermatoscope, controlled
  lighting). Ordinary phone photos are out of distribution and unreliable.
- HAM10000 skews toward European populations and lighter skin phototypes.
- Headline accuracy is inflated by `nv` being ~67% of the data; melanoma recall
  is the metric that actually matters.
