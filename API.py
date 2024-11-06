from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import pandas as pd
from supabase import create_client, Client
import torch
import base64

app = FastAPI()
torch.manual_seed(0)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("modelo.pt")

# Configuración de Supabase
url: str = "https://afwgthjhqrgxizqydmvs.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmd2d0aGpocXJneGl6cXlkbXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTU4Nzg4OTUsImV4cCI6MjAzMTQ1NDg5NX0.Oq0wjvVrT8YJ4Q3q7Ji8-28qljja8h1sEBzZV5oXzzc"
supabase: Client = create_client(url, key)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo de imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        # Obtener los resultados de detección
        results = model.predict(image_np, verbose=False, stream=True)
        results = list(results)

        # Convertir la imagen marcada a PIL antes de guardarla
        marked_image_np = results[0].plot()
        marked_image_pil = Image.fromarray(marked_image_np)
        buffered = io.BytesIO()
        marked_image_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Extraer las cajas de detección y conteos
        boxes_res = results[0].boxes.numpy()
        classes = boxes_res.cls
        count_1 = sum(classes)
        count_0 = len(classes) - count_1

        return JSONResponse(
            content={
                "positivos": int(count_1),
                "negativos": int(count_0),
                "image": img_base64
            }
        )

    except Exception as e:
        print("Error al procesar la imagen:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
