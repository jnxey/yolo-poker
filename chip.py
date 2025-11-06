import onnxruntime as ort
import numpy as np
import cv2
from ultralytics import YOLO

# ---------------- 配置 ----------------
ONNX_MODEL = "yolov8n.onnx"  # 已包含 NMS
IMG_PATH = "chips.jpg"
IMG_SIZE = 640

# 类别列表
CLASS_NAMES = list(YOLO("yolov8n.pt").names)

# 加载 ONNX
ort_session = ort.InferenceSession(ONNX_MODEL)

# 读取图片
img = cv2.imread(IMG_PATH)
h0, w0 = img.shape[:2]

# ---------------- 预处理 ----------------
scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
new_w, new_h = int(w0*scale), int(h0*scale)
img_resized = cv2.resize(img, (new_w, new_h))

# 填充到 IMG_SIZE x IMG_SIZE
img_padded = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 114/255.0
img_resized = img_resized.astype(np.float32)/255.0
img_padded[0:new_h, 0:new_w, :] = img_resized

# HWC -> CHW
img_input = np.transpose(img_padded, (2,0,1))
img_input = np.expand_dims(img_input, axis=0)

# ---------------- 推理 ----------------
inputs = {ort_session.get_inputs()[0].name: img_input}
outputs = ort_session.run(None, inputs)
pred = outputs[0]  # shape: (1, num_boxes, 6)

# ---------------- 处理输出 ----------------
pred = np.squeeze(pred)  # shape -> (num_boxes, 6)
if pred.ndim == 1:
    pred = pred.reshape(1, -1)

boxes = pred[:, :4]
scores = pred[:, 4]
class_ids = pred[:, 5].astype(int)

# 缩放回原图
boxes[:, [0,2]] /= scale
boxes[:, [1,3]] /= scale

# ---------------- 绘制 ----------------
for i in range(len(boxes)):
    box = boxes[i]
    x1, y1, x2, y2 = map(int, box)
    cls_id = class_ids[i]
    score = scores[i]
    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
    label = f"{cls_name} {score:.2f}"

    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(img, label, (x1, max(0,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)

cv2.imshow("YOLOv8 ONNX Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
