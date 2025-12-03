import fastapi
import onnxruntime as ort
import os
import json
import cv2
import numpy as np
from challenge.config import IMGSZ, MODEL_PATH_ONNX, CLASSES_PATH, CONF_THRES, IOU_THRES

app = fastapi.FastAPI()

# ===== Load model =====
session = None
class_names = []

# Load ONNX model
if os.path.exists(MODEL_PATH_ONNX):
    session = ort.InferenceSession(MODEL_PATH_ONNX, providers=["CPUExecutionProvider"])
    
# Load class names
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)["names"]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "ok",
        "model": "loaded" if session is not None else "unloaded",
        "classes": "loaded" if len(class_names) > 0 else "unloaded",
    }


@app.post("/predict", status_code=200)
async def post_predict(file: fastapi.UploadFile) -> dict:
    # Check model state
    if session is None:
        return fastapi.Response(status_code=400)
    
    # Read image bytes
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return fastapi.Response(status_code=415)  # Unsupported media type
    
    # Preprocess
    orig_h, orig_w = img.shape[:2]
    inp = cv2.resize(img, (IMGSZ, IMGSZ))
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...] # NCHW format

    # ONNX Inference
    out = session.run(None, {session.get_inputs()[0].name: inp})[0][0]

    # YOLOv11 ONNX outputs (C, N) -> we need (N, C)
    out = out.transpose()       # (8400, 21)

    # Split outputs
    boxes = out[:, :4]        # x,y, x,y
    classes = out[:, 4:]        # the rest is class logits

    # Confidence filtering
    # YOLOv11 -> confidence = max class probability
    cls_conf = np.max(classes, axis=1)        # (8400,)
    cls_ids  = np.argmax(classes, axis=1)     # (8400,)
    conf     = cls_conf                       # final confidence
    mask = conf > CONF_THRES

    boxes   = boxes[mask]
    conf    = conf[mask]
    cls_ids = cls_ids[mask]


    # NMS (pure OpenCV)
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=conf.tolist(),
        score_threshold=CONF_THRES,
        nms_threshold=IOU_THRES
    )

    if len(idxs) == 0:
        return {"detections": []}

    idxs = idxs.flatten()
    results = []
    scale_w = orig_w / IMGSZ
    scale_h = orig_h / IMGSZ

    for i in idxs:
        xc, yc, bw, bh = boxes[i]

        # XYWH -> XYXY conversion
        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        # Escalar a resolucion original
        x1 *= scale_w
        x2 *= scale_w
        y1 *= scale_h
        y2 *= scale_h

        # Limitar a dimensiones de la imagen
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        # resultados JSON
        results.append({
            "cls_id": int(cls_ids[i]),
            "cls_name": class_names[int(cls_ids[i])],
            "confidence": float(conf[i]),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }
        })

    return {"detections": results}
