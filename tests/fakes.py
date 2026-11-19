# Fake damage for robustness tests

import cv2
import numpy as np


def rotate_cropped(img: np.ndarray, deg: float) -> np.ndarray:
    # tilt and cut corners off
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=255)


def mud(img: np.ndarray, blobs: int, strength: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape
    layer = np.zeros(img.shape, np.float32)
    for _ in range(blobs):
        centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(80, 350)), int(rng.integers(60, 250)))
        cv2.ellipse(layer, centre, axes, float(rng.integers(0, 180)), 0, 360, 1.0, -1)
    layer = np.clip(cv2.GaussianBlur(layer, (0, 0), 25), 0, 1) * strength
    out = img.astype(np.float32) * (1 - layer) + 90 * layer + rng.normal(0, 8, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def perspective(img: np.ndarray) -> np.ndarray:
    # phone photo at an angle
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 30], [w - 10, 0], [w, h], [0, h - 40]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)


def draw_x(img: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.line(img, (cx - 15, cy - 15), (cx + 15, cy + 15), 30, 3)
    cv2.line(img, (cx - 15, cy + 15), (cx + 15, cy - 15), 30, 3)


def scribble(img: np.ndarray, box: tuple[int, int, int, int], seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    for _ in range(40):
        p1 = (int(rng.integers(x0 + 12, x1 - 12)), int(rng.integers(y0 + 12, y1 - 12)))
        p2 = (int(rng.integers(x0 + 12, x1 - 12)), int(rng.integers(y0 + 12, y1 - 12)))
        cv2.line(img, p1, p2, 20, 4)
