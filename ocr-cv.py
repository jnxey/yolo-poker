import cv2
import numpy as np


def extract_text_mask(gray):
    # 自适应阈值（对反光更稳）
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )

    # 去小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return bin_img


def deskew_chip_pca(img, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = extract_text_mask(gray)

    ys, xs = np.where(bin_img > 0)
    if len(xs) < 100:
        return img

    coords = np.column_stack((xs, ys)).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    vx, vy = eigenvectors[0]

    angle = np.degrees(np.arctan2(vy, vx))

    if debug:
        print(f"[PCA angle] {angle:.2f}")
        vis = img.copy()
        cx, cy = int(mean[0][0]), int(mean[0][1])
        cv2.line(
            vis,
            (cx - int(vx * 200), cy - int(vy * 200)),
            (cx + int(vx * 200), cy + int(vy * 200)),
            (0, 255, 0), 2
        )
        cv2.imshow("text_mask", bin_img)
        cv2.imshow("direction", vis)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


if __name__ == "__main__":
    img = cv2.imread("t22.png")
    assert img is not None, "图片没读到"

    fixed = deskew_chip_pca(img, debug=True)

    cv2.imshow("original", img)
    cv2.imshow("deskewed", fixed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
