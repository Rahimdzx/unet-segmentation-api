# Segmentation Service

Production-grade FastAPI service for U‑Net image segmentation powered by ONNX Runtime.

## Quick Start

```bash
git clone <repo>
cd segmentation_service_onnx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/unet_model_best.onnx weights/
uvicorn app.main:app --reload
```

POST an image to `/predict` and receive a base64‑encoded PNG mask.

## Exporting ONNX

```bash
python app/exporter.py --weights /path/to/unet_model_best.pth --out weights/unet_model_best.onnx
```

## Docker

```bash
docker build -t unet-api .
docker run -p 8000:8000 unet-api
```
