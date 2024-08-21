from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from supabase import create_client, Client

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Asegúrate de permitir los métodos necesarios
    allow_headers=["*"],
)


# Cargar el modelo
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

        # Convertir la imagen a formato adecuado para el modelo
        image_np = np.array(image)

        # Procesar la imagen con el modelo
        results = model(image_np)

        # Suponiendo que `results` tiene las propiedades `boxes` y `speed`
        # Extraer datos de los resultados
        boxes = results.pandas().xyxy[0]  # Aquí obtenemos las cajas como un DataFrame

        ki67_positivos = len(boxes[boxes['class'] == 0])  # Suponiendo clase 0 para positivos
        ki67_negativos = len(boxes[boxes['class'] == 1])  # Suponiendo clase 1 para negativos

        # Guardar en Supabase
        response = supabase.table("datos").insert({
            "ki67_positivos": ki67_positivos,
            "ki67_negativos": ki67_negativos,

        }).execute()

        if response.error:
            raise Exception(f"Error saving data: {response.error.message}")

        return JSONResponse(content={
            "ki67_positivos": ki67_positivos,
            "ki67_negativos": ki67_negativos,
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
