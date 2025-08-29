from ultralytics import YOLO

y8n = YOLO('yolov8n.pt')

y8n('./images/demo.jpg', show=True, save=True)

