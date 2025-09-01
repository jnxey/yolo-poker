from ultralytics import YOLO

# 加载预训练模型
y8n = YOLO('yolov8n.pt')

# y8n('./images/demo.jpg', show=True, save=True) # 单个图片分析

y8n.train(
    data='data.yaml',  # 数据集配置文件路径
    epochs=500,  # 训练轮次
    imgsz=640,  # 输入图片尺寸
    batch=32,  # 每次训练的批量
    device=0  # 训练方式 GPU=0，CPU='cpu'
)

print('模型训练完毕')
