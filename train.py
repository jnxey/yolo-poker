from ultralytics import YOLO # pip install ultralytics
import torch # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 版本号按系统的来

print("torch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))

# 加载预训练模型
y8n = YOLO('yolov8n.pt')

# y8n('./images/demo.jpg', show=True, save=True) # 单个图片分析

y8n.train(
    data='data.yaml',  # 数据集配置文件路径
    epochs=1,  # 训练轮次
    imgsz=640,  # 输入图片尺寸
    batch=32,  # 每次训练的批量
    device=0,  # 训练方式 GPU=0，CPU='cpu'
    amp=False,   # 关闭自动混合精度
    workers=0   # 关键：在 Windows 上建议先改成 0
)

print('模型训练完毕')
