"""FastAPI inference server for UdderNet.

Serves a minimal test frontend and two prediction endpoints backed by the
trained checkpoints in checkpoints/.

Usage:
    uv run uvicorn uddernet.app:app --reload
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

from uddernet.models import MastitisMLP, build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CLASS_NAMES = {0: "negative (healthy)", 1: "positive (mastitis)"}
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def image_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    state["device"] = device

    mlp_ckpt = CHECKPOINT_DIR / "mastitis_mlp.pt"
    if mlp_ckpt.exists():
        ckpt = torch.load(mlp_ckpt, map_location=device, weights_only=False)
        mlp = MastitisMLP(in_features=len(ckpt["feature_names"])).to(device)
        mlp.load_state_dict(ckpt["model_state"])
        mlp.eval()
        state["mlp"] = mlp
        state["feature_names"] = ckpt["feature_names"]
        state["scaler_mean"] = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        state["scaler_scale"] = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    cnn_ckpt = CHECKPOINT_DIR / "udder_cnn.pt"
    if cnn_ckpt.exists():
        ckpt = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        arch = ckpt.get("arch", "cnn")
        cnn = build_model(arch, num_classes=len(ckpt["classes"]),
                          **({"pretrained": False} if arch == "resnet18" else {}))
        cnn.load_state_dict(ckpt["model_state"])
        cnn.to(device).eval()
        state["cnn"] = cnn
        state["cnn_classes"] = ckpt["classes"]
        state["cnn_transform"] = image_transform(ckpt.get("img_size", 128))

    yield
    state.clear()


app = FastAPI(title="UdderNet", lifespan=lifespan)


class MilkSample(BaseModel):
    day: float = Field(..., description="Day of lactation")
    milk_temperature: float
    milk_ph: float
    milk_conductivity: float
    somatic_cell_count: float
    milk_yield: float
    clotting: int = Field(..., ge=0, le=1)

    def to_feature_dict(self) -> dict[str, float]:
        return {
            "Day": self.day,
            "Milk_Temperature": self.milk_temperature,
            "Milk_pH": self.milk_ph,
            "Milk_Conductivity": self.milk_conductivity,
            "Somatic_Cell_Count": self.somatic_cell_count,
            "Milk_Yield": self.milk_yield,
            "Clotting": float(self.clotting),
        }


def _tabular_mastitis_prob(sample: "MilkSample") -> float:
    """P(mastitis) from the tabular MLP."""
    features = sample.to_feature_dict()
    x = np.array([[features[name] for name in state["feature_names"]]], dtype=np.float32)
    x = (x - state["scaler_mean"]) / state["scaler_scale"]
    with torch.no_grad():
        logits = state["mlp"](torch.from_numpy(x).to(state["device"]))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])


def _image_mastitis_prob(image: Image.Image) -> float:
    """P(mastitis) from the image model (handles class ordering robustly)."""
    x = state["cnn_transform"](image).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        logits = state["cnn"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pos_idx = state["cnn_classes"].index("positive")
    return float(probs[pos_idx])


async def _read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image.")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "tabular_model_loaded": "mlp" in state,
        "image_model_loaded": "cnn" in state,
        "device": str(state.get("device", "unknown")),
    }


@app.post("/api/predict/tabular")
async def predict_tabular(sample: MilkSample):
    if "mlp" not in state:
        raise HTTPException(503, "Tabular model not trained yet. Run: uv run train-tabular")

    p_mastitis = _tabular_mastitis_prob(sample)
    pred = int(p_mastitis >= 0.5)
    return {
        "prediction": pred,
        "label": CLASS_NAMES[pred],
        "probability_mastitis": p_mastitis,
    }


@app.post("/api/predict/image")
async def predict_image(file: UploadFile):
    if "cnn" not in state:
        raise HTTPException(503, "Image model not trained yet. Run: uv run train-cnn")

    image = await _read_image(file)
    p_mastitis = _image_mastitis_prob(image)
    pred = int(p_mastitis >= 0.5)
    return {
        "prediction": CLASS_NAMES[pred],
        "probabilities": {"negative": 1.0 - p_mastitis, "positive": p_mastitis},
    }


@app.post("/api/predict/combined")
async def predict_combined(
    file: UploadFile,
    day: float = Form(...),
    milk_temperature: float = Form(...),
    milk_ph: float = Form(...),
    milk_conductivity: float = Form(...),
    somatic_cell_count: float = Form(...),
    milk_yield: float = Form(...),
    clotting: int = Form(...),
):
    """Late (decision-level) fusion of the image and tabular models.

    The image model leads: when it predicts *negative* we trust it more
    (0.7 image / 0.3 tabular); when it predicts *positive* we defer to the
    sensor data for confirmation (0.3 image / 0.7 tabular).
    """
    if "cnn" not in state:
        raise HTTPException(503, "Image model not trained yet. Run: uv run train-cnn")
    if "mlp" not in state:
        raise HTTPException(503, "Tabular model not trained yet. Run: uv run train-tabular")

    sample = MilkSample(
        day=day, milk_temperature=milk_temperature, milk_ph=milk_ph,
        milk_conductivity=milk_conductivity, somatic_cell_count=somatic_cell_count,
        milk_yield=milk_yield, clotting=clotting,
    )
    image = await _read_image(file)

    p_img = _image_mastitis_prob(image)
    p_tab = _tabular_mastitis_prob(sample)

    image_says_positive = p_img >= 0.5
    if image_says_positive:
        w_img, w_tab = 0.3, 0.7
    else:
        w_img, w_tab = 0.7, 0.3
    p_combined = w_img * p_img + w_tab * p_tab
    pred = int(p_combined >= 0.5)

    return {
        "prediction": pred,
        "label": CLASS_NAMES[pred],
        "probability_mastitis": p_combined,
        "weights": {"image": w_img, "tabular": w_tab},
        "components": {
            "image_probability_mastitis": p_img,
            "tabular_probability_mastitis": p_tab,
        },
    }
