from ultralytics import YOLO

def run():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        amp=False,
        workers=0
    )

if __name__ == "__main__":   # 必须加这句
    run()
