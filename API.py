from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
from ultralytics import YOLO

app = FastAPI()

# Configuración CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins= "https://webiacm-4rxkuqdb6-iaras-projects-d66e8430.vercel.app/",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLO
logging.info("Cargando modelo YOLO")
model = YOLO("modelo.pt")
logging.info("Modelo YOLO cargado correctamente")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Ejecutar el modelo YOLO
        results = model(image)
        
        # Inicializar contadores
        cant_pos = 0
        cant_neg = 0

        # Procesar resultados
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                if box.cls == 0:  # Clase 0 = positiva
                    cant_pos += 1
                else:  # Otras clases = negativas
                    cant_neg += 1
        
        # Anotar la imagen con las detecciones
        annotated_image = results[0].plot()
        annotated_image = Image.fromarray(annotated_image)

        # Convertir la imagen anotada a base64
        output_buffer = io.BytesIO()
        annotated_image.save(output_buffer, format="JPEG")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        # Crear la respuesta en formato JSON
        response_data = {
            "positivos": cant_pos,
            "negativos": cant_neg,
            "imagenProcesada": base64_image,
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
