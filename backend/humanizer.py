import cv2
import numpy as np


def apply_watercolor(img_cv):
    smooth = cv2.bilateralFilter(img_cv, 9, 75, 75)
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(smooth, 1.0, edges_bgr, 0.3, 0)


def apply_sketch(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    blurred_inv = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, blurred_inv, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_oil_painting(img_cv):
    try:
        return cv2.xphoto.oilPainting(img_cv, 7, 1)
    except AttributeError:
        result = img_cv.copy()
        for _ in range(4):
            result = cv2.bilateralFilter(result, 9, 150, 150)
        return result


def apply_flat_art(img_cv):
    shifted = cv2.pyrMeanShiftFiltering(img_cv, 10, 50)
    levels = 4
    step = 256 // levels
    posterized = (shifted // step) * step
    posterized = np.clip(posterized, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(posterized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    result = posterized.copy()
    result[edges > 0] = [0, 0, 0]
    return result


def apply_vintage(img_cv):
    img_float = img_cv.astype(np.float32) / 255.0
    r = img_float[:, :, 2]
    g = img_float[:, :, 1]
    b = img_float[:, :, 0]
    new_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 1)
    new_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 1)
    new_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 1)
    sepia_img = np.stack([new_b, new_g, new_r], axis=2)
    noise = np.random.normal(0, 0.03, sepia_img.shape).astype(np.float32)
    noisy = np.clip(sepia_img + noise, 0, 1)
    h, w = noisy.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1 - dist * 0.7, 0, 1)
    vignetted = noisy * vignette[:, :, np.newaxis]
    result_uint8 = (vignetted * 255).astype(np.uint8)
    hsv = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.6
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
