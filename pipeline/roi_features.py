import cv2
import numpy as np

def extract_features(gray_roi):
    roi = cv2.resize(gray_roi, (32, 32))

    mu = np.mean(roi)
    sigma = np.std(roi)

    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
    grad = np.mean(np.sqrt(gx**2 + gy**2))

    _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    compact = ecc = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True) + 1e-8
        compact = (4 * np.pi * area) / (peri * peri)

        if len(c) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(c)
            ecc = ma / (MA + 1e-8)

    return np.array([mu, sigma, grad, compact, ecc], dtype=np.float32)
