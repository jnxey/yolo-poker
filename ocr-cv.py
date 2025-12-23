import cv2
import numpy as np
import math

# 计算三点夹角
def angle(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    if a*b == 0:
        return 0
    return math.degrees(math.acos((a*a + b*b - c*c) / (2*a*b)))

# 判断是否为L形轮廓
def is_l_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    if len(approx) < 3:
        return False, None
    # 遍历所有三点组合
    pts = approx.reshape(-1, 2)
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i+1)%len(pts)]
        p3 = pts[(i+2)%len(pts)]
        ang = angle(p1, p2, p3)
        if 70 < ang < 110:  # 允许 ±20° 误差
            return True, (p1, p2, p3)
    return False, None

# 主函数
def detect_rotated_corner(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 边缘检测
    edges = cv2.Canny(bin_img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    l_corners = []
    for cnt in contours:
        ok, pts = is_l_shape(cnt)
        if ok:
            l_corners.append(pts)
            # 可视化角标
            for pt in pts:
                cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

    # 如果检测到至少2个角标，生成 ROI
    if len(l_corners) >= 2:
        all_points = np.array([pt for corner in l_corners for pt in corner])
        x_min = np.min(all_points[:,0])
        y_min = np.min(all_points[:,1])
        x_max = np.max(all_points[:,0])
        y_max = np.max(all_points[:,1])
        roi = img[y_min:y_max, x_min:x_max]
        cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
        cv2.imshow("ROI", roi)

    cv2.imshow("Detected Corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 测试
detect_rotated_corner("t30.png")
