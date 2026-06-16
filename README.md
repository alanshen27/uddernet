# UdderNet

**iGEM Le Rosey 2026** — early mastitis detection for dairy herds using two neural networks and late fusion.

Mastitis is one of the costliest diseases in dairy farming. UdderNet screens cows by combining **milk-sensor readings** and **udder photographs**, then fuses both predictions into a single diagnosis.

## The three models

### 1. MastitisMLP (tabular classifier)

A feed-forward neural network trained on `data/cow_milk_mastitis_dataset.csv` (800 labelled milk samples).

| Input features | Output |
|---|---|
| Day of lactation, milk temperature, pH, conductivity, somatic cell count, milk yield, clotting | P(mastitis) — a value between 0 and 1 |

Architecture: 7 → 64 → 32 → 2 (with batch normalisation and dropout). Inputs are standardised before inference.

```bash
uv run train-tabular
```

Checkpoint: `checkpoints/mastitis_mlp.pt`

### 2. ResNet-18 (image classifier)

A pretrained ResNet-18 (ImageNet weights) fine-tuned on ~600 udder images from the **iGEN** dataset, organised as:

```
data/images/
    positive/   # mastitis
    negative/   # healthy
```

Images are resized to 224×224, augmented during training (flips, rotation, colour jitter), and classified into two classes. The final layer is a 2-neuron head: one logit per class (negative / positive).

```bash
uv run train-cnn --model resnet18
```

Checkpoint: `checkpoints/udder_cnn.pt`

A custom from-scratch CNN (`UdderCNN`) is also available as a baseline via `--model cnn`.

### 3. Late fusion (combined prediction)

Because the image and sensor datasets are **not paired** (no cow ID linking a photo to a milk row), we fuse at the **decision level** rather than inside a single network.

Both models run independently. Their mastitis probabilities are blended with **image-led weighting**:

| Image prediction | Image weight | Sensor weight |
|---|---|---|
| Negative (healthy) | 0.7 | 0.3 |
| Positive (mastitis) | 0.3 | 0.7 |

```
P_fused = w_image × P_image + w_sensor × P_sensor
```

Rationale: a healthy-looking udder is strong evidence of health (trust the image); a suspicious image should be confirmed by sensor data before flagging mastitis.

Endpoint: `POST /api/predict/combined`

## Project layout

```
data/
    cow_milk_mastitis_dataset.csv
    images/positive/
    images/negative/
src/uddernet/
    models.py            # MastitisMLP, UdderCNN, build_resnet18
    train_tabular.py
    train_cnn.py
    app.py               # FastAPI inference server
    static/index.html    # phone-style onboarding UI
docs/report.md           # academic write-up
checkpoints/             # saved models (created on training)
```

## Setup

```bash
uv sync
```

## Training

```bash
uv run train-tabular                  # MLP on the CSV
uv run train-cnn --model resnet18     # ResNet-18 on images
```

Both accept `--epochs`, `--batch-size`, `--lr`. The best checkpoint (by validation accuracy) is saved automatically.

## Run the app

```bash
uv run uvicorn uddernet.app:app --reload
```

Open http://127.0.0.1:8000 — a phone-style onboarding flow walks you through sensor entry, image upload, and the fused diagnosis.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Which models are loaded |
| `POST` | `/api/predict/tabular` | Sensor data only |
| `POST` | `/api/predict/image` | Image only |
| `POST` | `/api/predict/combined` | Late fusion of both |

## iGEM Le Rosey 2026

UdderNet is developed as part of the **iGEM Le Rosey 2026** synthetic biology project, targeting improved dairy herd health through automated mastitis screening. The dual-modality approach mirrors how a farmer would assess a cow: look at the udder *and* check the milk.

For the full technical report (data analysis, architecture details, results, limitations), see [`docs/report.md`](docs/report.md).
