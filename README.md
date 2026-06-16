# UdderNet

Mastitis detection in dairy cows with two PyTorch models:

- **`UdderCNN`** — convolutional network for udder images
- **`MastitisMLP`** — classification network for tabular milk-sensor data

See [`docs/report.md`](docs/report.md) for the full write-up (data, architectures, results).

## Project layout

```
data/
    cow_milk_mastitis_dataset.csv   # tabular milk-sensor dataset
    images/positive/                # mastitis images (add your own)
    images/negative/                # healthy images
src/uddernet/
    models.py                       # UdderCNN + MastitisMLP definitions
    train_cnn.py                    # image training pipeline
    train_tabular.py                # tabular training pipeline
    app.py                          # FastAPI inference server
    static/index.html               # browser test frontend
docs/report.md                      # academic report
checkpoints/                        # saved models (created on first training run)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Training

```bash
uv run train-tabular                # tabular MLP on the CSV (works out of the box)
uv run train-cnn                    # image CNN, once images are in data/images/
```

Both commands accept `--epochs`, `--batch-size`, `--lr`; the best checkpoint (by
validation accuracy) is saved to `checkpoints/`.

## Test frontend

Start the inference server from the project root:

```bash
uv run uvicorn uddernet.app:app --reload
```

Then open http://127.0.0.1:8000 — the page has a form for milk-sensor values
(tabular network) and an image upload (CNN). Models are loaded from
`checkpoints/` at startup, so train at least one model first.

API endpoints: `GET /api/health`, `POST /api/predict/tabular`, `POST /api/predict/image`.
