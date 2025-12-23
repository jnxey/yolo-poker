import cv2
import numpy as np
import math

# 1. 读取图片
image = cv2.imread('t31.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. 蓝色范围
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 可选：去噪
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 3. 找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tri_centers = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 3 and 50 < area < 5000:  # 三角形
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        tri_centers.append((cx, cy))
        cv2.drawContours(image, [approx], 0, (0,255,0), 2)

if len(tri_centers) != 2:
    print("未找到两个蓝色三角形")
    cv2.imshow("mask", mask)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    exit()

# 4. 计算旋转角度
(x1, y1), (x2, y2) = tri_centers
angle_rad = math.atan2(y2 - y1, x2 - x1)
angle_deg = math.degrees(angle_rad)

# 5. 旋转图片
h, w = image.shape[:2]
center = ((x1+x2)//2, (y1+y2)//2)
M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

# 6. 裁剪 code 区域
padding = 10
x_min = min(x1, x2) - padding
x_max = max(x1, x2) + padding
y_min = min(y1, y2) - padding
y_max = max(y1, y2) + 2*padding  # code 在三角形上方
code_crop = rotated[y_min:y_max, x_min:x_max]

# 显示结果
cv2.imshow("mask", mask)
cv2.imshow("triangles", image)
cv2.imshow("rotated", rotated)
cv2.imshow("code", code_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
