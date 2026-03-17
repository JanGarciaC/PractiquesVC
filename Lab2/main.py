import os
import cv2
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "fotos", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "fotos", "output")

def crop_borders(img, percent=0.1):
    h, w = img.shape
    dh = int(h * percent)
    dw = int(w * percent)
    return img[dh:h-dh, dw:w-dw]

def split_channels(img):
    h = img.shape[0] // 3
    return img[0:h], img[h:2*h], img[2*h:3*h]

def shift_image(img, dx, dy):
    h, w = img.shape
    result = np.zeros_like(img)

    x1 = max(0, dx)
    x2 = min(h, h + dx)
    y1 = max(0, dy)
    y2 = min(w, w + dy)

    result[x1:x2, y1:y2] = img[x1-dx:x2-dx, y1-dy:y2-dy]
    return result


def ncc(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a -= np.mean(a)
    b -= np.mean(b)

    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))
    if denom == 0:
        return -1

    return np.sum(a*b) / denom


"""
def ssd(a, b):
    return -np.sum((a - b) ** 2)
"""


def phase_correlation(a, b):
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)

    R = A * np.conj(B)
    R /= np.abs(R) + 1e-8

    r = np.fft.ifft2(R)
    r = np.abs(r)

    dx, dy = np.unravel_index(np.argmax(r), r.shape)

    if dx > a.shape[0] // 2:
        dx -= a.shape[0]
    if dy > a.shape[1] // 2:
        dy -= a.shape[1]

    return dx, dy


def find_shift(ref, target, max_shift=15):
    best_score = -1
    best_shift = (0, 0)

    ref_c = crop_borders(ref)

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted = shift_image(target, dx, dy)
            shifted_c = crop_borders(shifted)

            score = ncc(ref_c, shifted_c)

            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    return best_shift


def pyramid_align(ref, target, levels=4):
    if levels == 0 or min(ref.shape) < 64:
        return find_shift(ref, target, 15)

    ref_small = cv2.pyrDown(ref)
    target_small = cv2.pyrDown(target)

    dx, dy = pyramid_align(ref_small, target_small, levels-1)

    dx *= 2
    dy *= 2

    best_shift = (dx, dy)
    best_score = -1

    ref_c = crop_borders(ref)

    for i in range(-2, 3):
        for j in range(-2, 3):
            sx = dx + i
            sy = dy + j

            shifted = shift_image(target, sx, sy)
            shifted_c = crop_borders(shifted)

            score = ncc(ref_c, shifted_c)

            if score > best_score:
                best_score = score
                best_shift = (sx, sy)

    return best_shift


def align_channels(b, g, r):
    ref = g

    shift_b = pyramid_align(ref, b)
    shift_r = pyramid_align(ref, r)

    b_aligned = shift_image(b, *shift_b)
    r_aligned = shift_image(r, *shift_r)

    h = min(b_aligned.shape[0], g.shape[0], r_aligned.shape[0])
    w = min(b_aligned.shape[1], g.shape[1], r_aligned.shape[1])

    return b_aligned[:h, :w], g[:h, :w], r_aligned[:h, :w]


def crop_valid_region(b, g, r):
    mask = (b > 0) & (g > 0) & (r > 0)
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        return b, g, r

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return b[y0:y1, x0:x1], g[y0:y1, x0:x1], r[y0:y1, x0:x1]


def enhance(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_image(path):
    start = time.time()

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0

    b, g, r = split_channels(img)
    b, g, r = align_channels(b, g, r)
    b, g, r = crop_valid_region(b, g, r)

    b = crop_borders(b, 0.05); g = crop_borders(g, 0.05); r = crop_borders(r, 0.05)

    color = np.dstack((b, g, r))

    elapsed = time.time() - start

    color = enhance(color)

    return color, elapsed


def save_image(img, input_path):
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, name + "_color.jpg")
    cv2.imwrite(output_path, img)


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(INPUT_DIR, file)

            print(f"Processant: {file}")

            img, t = process_image(path)

            if img is not None:
                save_image(img, path)
                print(f"Temps per {file}: {t:.3f} s")
