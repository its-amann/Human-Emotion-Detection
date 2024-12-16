from typing import Union
from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt
import os
'''
------------------------ changed location of model from onnx_infrenece to main ------------------------
'''

app = FastAPI(title="Emotion Detection API", version="0.1")
app.include_router(main_router)

# Get absolute path to the ONNX model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'vit_onnx.onnx')

# Initialize model globally
m = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

@app.get("/")
def root():
    return {"message": "Welcome to Emotion Detection API"}
