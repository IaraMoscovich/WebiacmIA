from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import pandas as pd
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

model = YOLO("modelo.pt")

# Configuración de Supabase
url: str = "https://afwgthjhqrgxizqydmvs.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmd2d0aGpocXJneGl6cXlkbXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTU4Nzg4OTUsImV4cCI6MjAzMTQ1NDg5NX0.Oq0wjvVrT8YJ4Q3q7Ji8-28qljja8h1sEBzZV5oXzzc"
supabase: Client = create_client(url, key)

def boxes_to_dataframe(boxes):
    # Verificar si hay cajas detectadas
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=['class'])

    classes = boxes.cls.tolist()

    # Crear un DataFrame
    df = pd.DataFrame({
        'class': classes
    })

    return df

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo de imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        results = model.predict(image_np)

        # Convertir los resultados a un DataFrame
        boxes_df = boxes_to_dataframe(results[0].boxes)
        

        # Convertir el DataFrame a un JSON serializable
        boxes_json = boxes_df.to_json(orient='records')
        print(boxes_json)

        return JSONResponse(content={"detections": boxes_json})

    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
