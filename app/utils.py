import numpy as np
from PIL import Image

def preprocess(image, size=(512, 512)):
    image = image.resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = arr[None]
    return arr

def postprocess(pred):
    prob = pred[0, 0]
    mask = (prob > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)
