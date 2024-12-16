from fastapi import APIRouter,UploadFile,File,HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_infrence import emotion_detector
from service.core.schemas.input import OutputSchema
emotion_router = APIRouter()

@emotion_router.post("/detect")
async def detect_emotion(im : UploadFile):

    if im.filename.split('.')[-1] not in ['jpg','jpeg','png']:
        raise HTTPException (status_code = 415,detail = 'Unsupported Media Type')
    img = Image.open(BytesIO(im.file.read()))
    img = np.array(img)

    return  emotion_detector(img)