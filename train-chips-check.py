import os
import math
from pathlib import Path

# ===================== 配置区 =====================
LABEL_DIR = r"D:\Work\yolo-poker\chips-angle\labels\train_bak"  # 原6列标签文件夹
OUTPUT_DIR = r"D:\Work\yolo-poker\chips-angle\labels\train"  # 输出4点标签文件夹
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def angle_to_points(xc, yc, w, h, angle):
    """
    将中心+宽高+旋转角度转换为四点坐标
    xc, yc, w, h: 归一化坐标
    angle: 弧度
    返回: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    # 先缩放到图片单位假设图片为 1x1
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # 四点相对中心的偏移
    dx = w / 2
    dy = h / 2
    # 原始矩形四个角点
    corners = [
        (-dx, -dy),  # 左上
        ( dx, -dy),  # 右上
        ( dx,  dy),  # 右下
        (-dx,  dy)   # 左下
    ]
    points = []
    for px, py in corners:
        x = xc + px * cos_a - py * sin_a
        y = yc + px * sin_a + py * cos_a
        points.append((x, y))
    return points

# 遍历所有标签文件
for txt_file in Path(LABEL_DIR).glob("*.txt"):
    output_file = Path(OUTPUT_DIR) / txt_file.name
    lines_out = []
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 6:
                print(f"跳过 {txt_file}: 非6列")
                continue
            cls, xc, yc, w, h, angle = parts
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            angle = float(angle)
            points = angle_to_points(xc, yc, w, h, angle)
            # flatten为 x1 y1 x2 y2 x3 y3 x4 y4
            points_flat = [str(round(v, 6)) for p in points for v in p]
            line_out = " ".join([cls] + points_flat)
            lines_out.append(line_out)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_out))

print(f"✅ 转换完成，4点标签保存在: {OUTPUT_DIR}")
