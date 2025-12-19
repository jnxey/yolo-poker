from paddleocr import PaddleOCR

ocr = PaddleOCR(
    det_model_dir=r"C:\Users\李德锋\.paddlex\official_models\PP-LCNet_x1_0_doc_ori",
    rec_model_dir=r"C:\Users\李德锋\.paddlex\official_models\PP-LCNet_x1_0_doc_ori",
    use_angle_cls=False,  # 是否使用方向分类
    lang='en'  # 根据你训练的语言选择
)

print("Det model used:", ocr.det_model_dir)
print("Rec model used:", ocr.rec_model_dir)

img_path = "t22.png"
result = ocr.ocr(img_path, cls=True)

for line in result:
    print(line)
