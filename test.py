from ultralytics import YOLO # pip install ultralytics

# 加载预训练模型
# 加载预训练模型
# model = YOLO('yolov8n.pt')
model = YOLO('./runs/detect/train/weights/best.pt')

model('./pokers/test/t4.jpg', show=True, save=True) # 单个图片分析

print('测试完毕')
