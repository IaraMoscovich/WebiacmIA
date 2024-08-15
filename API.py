from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info("Importing Ultralytics YOLO")

from ultralytics import YOLO

logging.info("Loading YOLO modelo.pt")
model = YOLO("modelo.pt")
logging.info("YOLO modelo.pt Loaded")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        
        # Process the image (for demonstration purposes, we'll just get its format)
        image_format = model(image)
        print(image_format)
        
        # Return a string response
        return JSONResponse(content={"message": f"{image_format}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

import uvicorn
    #uvicorn.run(app, host="127.0.0.1", port=8000)
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
