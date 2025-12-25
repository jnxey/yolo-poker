import cv2
import numpy as np
import easyocr
import re

# 初始化 OCR（建议全局只初始化一次）
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# 合法面值（按你的筹码调整）
DENOMS = {'1', '5', '10', '25', '50', '100', '200', '500', '1000', '2000', '5000', '10000', '20000', '50000', '100000',
          '200000', '500000'}
# 码长度
CODE_LEN = 6


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def is_code(text, min_len=6):
    """
    判断是否像 code（固定长度）
    """
    text = text.replace(' ', '')
    return len(text) == CODE_LEN


def find_value_and_code(image, conf_thresh=0.85):
    results = reader.readtext(image, detail=1)
    denom = None
    code = None
    for bbox, txt, conf in results:
        t = txt.strip().upper().replace(' ', '')
        if conf < conf_thresh:
            print(f"   ✗ TEXT   : {t} (conf={conf:.2f}) [LOW_CONF]")
            continue  # 置信度过滤
        # 面值判断
        if t in DENOMS:
            denom = t
            print(f"   ✓ DENOM  : {t} (conf={conf:.2f})")
        # code 判断
        elif is_code(t):
            code = t
            print(f"   ✓ CODE   : {t} (conf={conf:.2f})")
        else:
            print(f"   • TEXT   : {t} (conf={conf:.2f}) [IGNORED]")
    if denom and code:
        return denom, code
    return None, None



def recognize_chip(image_path, step=10):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图片读取失败")
    for angle in range(0, 181, step):
        print(f"-------------角度{angle}-----------------")
        rotated = rotate_image(img, -angle)
        denom, code = find_value_and_code(rotated)
        if denom and code:
            return {
                'angle': angle,
                'denomination': denom,
                'code': code
            }
    return None

result = recognize_chip("t27.png")

if result:
    print("识别成功：", result)
else:
    print("识别失败")
