
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from torchvision import transforms

from model import MNISTCNN

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pt")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = MNISTCNN()
    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model unavailable")

    raw = await file.read()

    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(413, "Image too large")

    try:
        image = Image.open(io.BytesIO(raw)).convert("L")
        image = ImageOps.fit(image, (28, 28))

        # MNIST uses bright digits on dark backgrounds.
        tensor = transforms.ToTensor()(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            digit = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, digit].item())

        return {
            "prediction": digit,
            "confidence": round(confidence, 4)
        }

    except (OSError, ValueError) as exc:
        raise HTTPException(400, "Invalid image") from exc
