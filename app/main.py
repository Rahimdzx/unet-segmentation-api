from fastapi import FastAPI, UploadFile, File
import io, base64, numpy as np
from PIL import Image
import onnxruntime as ort
from .utils import preprocess, postprocess

_session = None

def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession("weights/unet_model_best.onnx", providers=["CPUExecutionProvider"])
    return _session

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(image)
    inputs = {get_session().get_inputs()[0].name: x}
    y = get_session().run(None, inputs)[0]
    mask = postprocess(y)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"mask": encoded}
