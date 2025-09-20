from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
from pneumonia_model import PneumoniaModel
from config import *
import os
import time

model_path = "trained/Resnet50Model2-adam-0.0001-64-20-augment/epoch_19/model.pth"
app = FastAPI(title="Pneumonia Detection API")
model = PneumoniaModel(model_path=model_path)

temp_folder = os.path.join("api", "tmp")
os.makedirs(temp_folder, exist_ok=True)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file

    temp_path = os.path.join(temp_folder, file.filename or f"{time.time()}.png")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    visualization, pred_label, original_image = model.predict(temp_path)
    return {"prediction": CLASSES[pred_label]}  # type: ignore


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
