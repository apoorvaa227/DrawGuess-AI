import base64
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
from torchvision import models, transforms


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "runs" / "doodle_classifier" / "best_model.pt"
LABELS_PATH = ROOT / "runs" / "doodle_classifier" / "labels.json"
IMG_SIZE = 96
BG_VALUE = 255
CONTENT_THRESHOLD = 245
EXCLUDED_LABELS = {
    "line",
    "circle",
    "square",
    "triangle",
    "squiggle",
    "zigzag",
}


class PredictRequest(BaseModel):
    image: str
    topk: int = 5


app = FastAPI(title="Doodle Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
labels = []
label_to_idx = {}
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


def load_artifacts():
    global model, labels, label_to_idx
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()


def decode_base64_image(payload: str) -> Image.Image:
    if "," not in payload:
        raise ValueError("Expected a data URL image payload.")

    _, encoded = payload.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Keep raw tones close to training PNG style (white background, dark strokes).
    return image


def preprocess_for_model(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    # Mark foreground doodle pixels by detecting dark strokes on white background.
    binary = grayscale.point(lambda px: 255 if px < CONTENT_THRESHOLD else 0)
    bbox = binary.getbbox()

    if bbox is None:
        # Empty drawing fallback: still provide valid image tensor.
        return grayscale.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    cropped = grayscale.crop(bbox)
    w, h = cropped.size
    side = max(w, h)

    # Center doodle inside a square canvas with original dark background.
    square = Image.new("L", (side, side), color=BG_VALUE)
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    square.paste(cropped, (paste_x, paste_y))

    # Small uniform border helps preserve edge strokes during resize.
    bordered = ImageOps.expand(square, border=max(2, side // 12), fill=BG_VALUE)
    return bordered.resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR).convert("RGB")


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "classes": len(labels)}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        image = decode_base64_image(request.image)
        image = preprocess_for_model(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        for label in EXCLUDED_LABELS:
            idx = label_to_idx.get(label)
            if idx is not None:
                logits[:, idx] = -1e9
        probs = F.softmax(logits, dim=1).squeeze(0)

    topk = max(1, min(request.topk, len(labels)))
    values, indices = torch.topk(probs, k=topk)

    predictions = [
        {"label": labels[idx], "score": float(score)}
        for score, idx in zip(values.tolist(), indices.tolist())
    ]

    return {"predictions": predictions}
