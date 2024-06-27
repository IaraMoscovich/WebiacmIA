from typing import Optional
from typing import Union
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()
model = YOLO("modelo.pt")

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/image")
#Aca los datos serian la direccion del .png a analizar
async def ia(datos): #chequear que ia datos agarra el directorio de la imagen no la imagen en si
    results = model(datos)
    return {"resultado": results}