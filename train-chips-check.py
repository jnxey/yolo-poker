# import math
# from pathlib import Path
#
# bad = []
#
# for p in Path("chips-angle/labels/train").rglob("*.txt"):
#     for i, line in enumerate(p.read_text().splitlines()):
#         parts = line.strip().split()
#         if len(parts) != 6:
#             bad.append((p, i+1, "列数错误", line))
#             continue
#
#         try:
#             angle = float(parts[5])
#         except:
#             bad.append((p, i+1, "angle 非数字", line))
#             continue
#
#         if not (-math.pi/2 <= angle < math.pi/2):
#             bad.append((p, i+1, "angle 超范围", angle))
#
# if bad:
#     for b in bad[:20]:
#         print(b)
# else:
#     print("✔ 所有 angle 合法")

# from pathlib import Path
#
# for p in Path("chips-angle/labels/train").rglob("*.txt"):
#     for line in p.read_text().splitlines():
#         parts = line.split()
#         if len(parts) != 6:
#             continue
#         nums = list(map(float, parts[1:5]))
#         if any(v <= 0 or v > 1 for v in nums):
#             print("坐标非法:", p, line)
#             raise SystemExit
# print("✔ 坐标全部在 0~1")

# from pathlib import Path
#
# bad = 0
# total = 0
#
# for p in Path("chips-angle/labels/train").rglob("*.txt"):
#     for line in p.read_text().splitlines():
#         cls, x, y, w, h, a = line.split()
#         w, h = float(w), float(h)
#         total += 1
#         if w < h:
#             bad += 1
#
# print(f"总标注数: {total}")
# print(f"w < h 的标注数: {bad}")

# import math
# from pathlib import Path
#
# def fix_obb(label_dir):
#     for p in Path(label_dir).rglob("*.txt"):
#         fixed = []
#         for line in p.read_text().splitlines():
#             cls, x, y, w, h, a = line.split()
#             w, h, a = float(w), float(h), float(a)
#
#             if w < h:
#                 w, h = h, w
#                 a += math.pi / 2
#
#             # 归一化 angle 到 [-pi/2, pi/2)
#             while a < -math.pi/2:
#                 a += math.pi
#             while a >= math.pi/2:
#                 a -= math.pi
#
#             fixed.append(f"{cls} {x} {y} {w} {h} {a}")
#
#         p.write_text("\n".join(fixed))
#
# # 修改下面路径
# fix_obb("chips-angle/labels/train")
# # fix_obb("labels/val")


from pathlib import Path

img_dir = Path("D:/Work/yolo-poker/chips-angle/images/train")
imgs = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
print("训练图片数量:", len(imgs))

label_dir = Path("D:/Work/yolo-poker/chips-angle/labels/train")
missing = []

for img in imgs:
    lbl = label_dir / (img.stem + ".txt")
    if not lbl.exists():
        missing.append(img.name)

print("缺失 label 的图片:", missing[:20])




