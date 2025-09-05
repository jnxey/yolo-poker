from ultralytics import YOLO # pip install ultralytics

# 加载预训练模型
# 加载预训练模型
# model = YOLO('yolov8n.pt')
model = YOLO('./runs/detect/train/weights/best.pt')

# 单张图片推理
results = model.predict(
    data='data.yaml',
    source='./pokers/test/17.png',  # 确保路径存在，尽量避免中文目录
    conf=0.25,
    device='cpu',
    save=True
)

# 遍历结果
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    print("检测框:", boxes)
