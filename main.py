from typing import Optional
from typing import Union
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()
model = YOLO("modelo.pt")

@app.post("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/upload-image")
#Aca los datos serian la direccion del .png a analizar
async def ia(datos): #chequear que ia datos agarra el directorio de la imagen no la imagen en si
    print(datos)
    results = model(datos)
    return {"resultado": results}

#PARA CORRER LOCAL:python -m uvicorn API:app --host 0.0.0.0 --port 8000 --reload