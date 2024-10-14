from kidney_Disease_classifier.pipeline.prediction import PredictionPipeline
from kidney_Disease_classifier.utils.common import decodeImage
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


import os


app =FastAPI()

origins = ['0.0.0.0',
   "http://{0.0.0.0}:{8000}"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train():
        os.system("python main.py")

        return "Training Completed"

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    image = body["image"]
    image = decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return JSONResponse(result)

if __name__ == "__main__":
     clApp = ClientApp()
     uvicorn.run(app, host="0.0.0.0", port=8000)

