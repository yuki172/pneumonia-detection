from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
from pneumonia_model import PneumoniaModel
from config import *
import os
import time
from grad_cam import save_grad_cam_figs
from PIL import Image
import io
import base64
import numpy as np


model_path = "trained/Resnet50Model2-adam-0.0001-64-20-augment/epoch_19/model.pth"
app = FastAPI(title="Pneumonia Detection API")
model = PneumoniaModel(model_path=model_path)


def numpy_to_base64(img_array: np.ndarray) -> str:
    image = (
        Image.fromarray((img_array * 255).astype(np.uint8))
        if img_array.dtype == np.float32
        else Image.fromarray(img_array)
    )
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    visualization, pred_label, original_image = model.predict(image)

    visualization_base64 = numpy_to_base64(visualization)
    original_image_base64 = numpy_to_base64(original_image)

    return {"prediction": CLASSES[pred_label], "visualization": visualization_base64, "original_image": original_image_base64}  # type: ignore


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
