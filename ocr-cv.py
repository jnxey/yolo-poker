import cv2
import numpy as np
import math

# -----------------------
# 1. 读取图片
# -----------------------
image = cv2.imread('t31.png')
h, w = image.shape[:2]

# -----------------------
# 2. 转换 HSV，提取蓝色三角形
# -----------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 蓝色范围，可根据实际调整
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 去噪
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

cv2.imshow("blue_mask", mask)

# -----------------------
# 3. 找轮廓，筛选三角形
# -----------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tri_centers = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50 or area > 5000:  # 过滤面积
        continue
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:  # 三角形
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        tri_centers.append((cx, cy))
        cv2.drawContours(image, [approx], 0, (0,255,0), 2)

cv2.imshow("triangles_detected", image)

if len(tri_centers) != 2:
    print("未找到两个蓝色三角形")
    cv2.waitKey(0)
    exit()

# -----------------------
# 4. 计算旋转角度
# -----------------------
(x1, y1), (x2, y2) = tri_centers
angle_rad = math.atan2(y2 - y1, x2 - x1)
angle_deg = math.degrees(angle_rad)

# -----------------------
# 5. 确认三角形在图片下方
# -----------------------
tri_avg_y = (y1 + y2) / 2
if tri_avg_y > h / 2:
    # 如果三角形在上半部分，需要旋转180度
    angle_deg += 180

# -----------------------
# 6. 旋转图片
# -----------------------
center = ((x1 + x2)//2, (y1 + y2)//2)
M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated", rotated)

# -----------------------
# 7. 裁剪 code 区域
# -----------------------
# padding_x = 10
# padding_y = 10
# x_min = min(x1, x2) - padding_x
# x_max = max(x1, x2) + padding_x
# y_min = min(y1, y2) - 2*padding_y  # code 在三角形上方
# y_max = max(y1, y2) + padding_y
#
# # 防止超出图片边界
# x_min = max(0, x_min)
# x_max = min(w, x_max)
# y_min = max(0, y_min)
# y_max = min(h, y_max)
#
# code_crop = rotated[y_min:y_max, x_min:x_max]
# cv2.imshow("code_crop", code_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
