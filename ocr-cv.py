import cv2
import numpy as np
import math

# 1. 读取图片
image = cv2.imread('t30.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # 假设三角形是深色

# 2. 找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tri_centers = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 3 and 50 < area < 5000:  # 三角形且面积合理
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        tri_centers.append((cx, cy))
        cv2.drawContours(image, [approx], 0, (0,255,0), 2)

# 确保只取两个三角形
if len(tri_centers) != 2:
    print("未找到两个三角形")
    exit()

(x1, y1), (x2, y2) = tri_centers

# 3. 计算旋转角度
angle_rad = math.atan2(y2 - y1, x2 - x1)
angle_deg = math.degrees(angle_rad)

# 4. 旋转图片
h, w = image.shape[:2]
center = ((x1+x2)//2, (y1+y2)//2)
M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

# 5. 裁剪 code 区域
padding = 10  # 可调整
x_min = min(x1, x2) - padding
x_max = max(x1, x2) + padding
y_min = min(y1, y2) - padding
y_max = max(y1, y2) + 2*padding  # code 在三角形上方

code_crop = rotated[y_min:y_max, x_min:x_max]

# 显示结果
cv2.imshow("triangles", image)
cv2.imshow("rotated", rotated)
cv2.imshow("code", code_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
