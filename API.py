from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# Configuración de CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = YOLO("modelo.pt")

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

        # Extraer datos de los resultados
        boxes = results.boxes  # Objeto Boxes con las coordenadas y más
        names = results.names  # Diccionario de nombres de clases
        orig_img = image_np    # Imagen original en formato numpy array
        orig_shape = results.orig_shape  # Forma original de la imagen
        speed = results.speed  # Tiempos de procesamiento

        # Convertir cajas y otros datos a formatos serializables
        boxes_data = []
        for box in boxes:
            box_data = {
                "coordinates": box.xyxy.tolist(),  # Coordenadas de la caja
                "class_id": box.cls.tolist(),       # ID de la clase
                "confidence": box.conf.tolist()      # Puntaje de confianza
            }
            boxes_data.append(box_data)

        # Retornar datos como JSON
        return JSONResponse(content={
            "boxes": boxes_data,
            "names": names,
            "orig_shape": orig_shape,
            "speed": speed
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    # Ejecutar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
