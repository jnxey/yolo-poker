from ultralytics import YOLO # pip install ultralytics
import time

# 加载预训练模型
# model = YOLO('yolov8n.pt')
model = YOLO('./chips-best8m.pt')

# 单张图片推理
t1 = time.time()
results = model.predict(
    data='data-chips.yaml',
    source='./t21.jpg',  # 确保路径存在，尽量避免中文目录
    conf=0.3,
    device='cpu',
    save=True,
    show=False
)
print("YOLO耗时:", time.time()-t1)

# 遍历结果
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    print("检测框:", boxes)
