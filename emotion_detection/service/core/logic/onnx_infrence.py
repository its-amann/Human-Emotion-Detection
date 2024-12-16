import onnxruntime as rt
import numpy as np
import cv2
import os
import time
import service.main as s


def emotion_detector(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    time_init = time.time()

    # Get absolute path to the ONNX model

    
    # Process image
    img = cv2.resize(img, (256,256))
    img = np.float32(img)
    img = np.expand_dims(img, axis=0)

    # Run inference
    input_name = s.m.get_inputs()[0].name
    output_name = s.m.get_outputs()[0].name
    onnx_pred = s.m.run([output_name], {input_name: img})
    
    time_elapsed = time.time() - time_init

    # Get emotion
    if np.argmax(onnx_pred[0][0]) == 0:
        emotion = "Angry"
    elif np.argmax(onnx_pred[0][0]) == 1:
        emotion = "sad"
    elif np.argmax(onnx_pred[0][0]) == 2:
        emotion = "happy"

    return {"emotion": emotion,
            "time_elapsed": str(time_elapsed),}