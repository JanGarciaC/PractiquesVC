import argparse
import os
import sys

import cv2
import numpy as np


def split_channels(img):
    """Split a Prokudin-Gorskii plate into 3 channels.
    The plate typically contains three grayscale bands in vertical or horizontal order.
    """
    h, w = img.shape[:2]
    if h % 3 == 0 and h // 3 > 0:
        step = h // 3
        b = img[0:step, :]
        g = img[step:2*step, :]
        r = img[2*step:3*step, :]
        return b, g, r

    if w % 3 == 0 and w // 3 > 0:
        step = w // 3
        b = img[:, 0:step]
        g = img[:, step:2*step]
        r = img[:, 2*step:3*step]
        return b, g, r

    raise ValueError("Input image cannot be split into 3 equal channels. "
                     "Ensure the input is a Prokudin-Gorskii plate with 3 stacked channels.")


def crop_image(img, crop=20):
    if crop <= 0:
        return img
    h, w = img.shape[:2]
    return img[crop:h-crop, crop:w-crop]


def ssd_loss(fixed, moving):
    diff = fixed.astype(np.float32) - moving.astype(np.float32)
    return np.mean(diff * diff)


def shift_and_crop(moving, dy, dx):
    h, w = moving.shape[:2]
    shifted = np.zeros_like(moving)
    if dy >= 0:
        y0_src, y0_dst = 0, dy
        y_len = h - dy
    else:
        y0_src, y0_dst = -dy, 0
        y_len = h + dy

    if dx >= 0:
        x0_src, x0_dst = 0, dx
        x_len = w - dx
    else:
        x0_src, x0_dst = -dx, 0
        x_len = w + dx

    shifted[y0_dst:y0_dst + y_len, x0_dst:x0_dst + x_len] = moving[y0_src:y0_src + y_len, x0_src:x0_src + x_len]
    return shifted


def align_pair(base, moving, max_shift=30, crop_border=20, initial_shift=(0, 0)):
    best_score = float('inf')
    best_shift = initial_shift

    base_crop = crop_image(base, crop_border)
    moving_crop_template = crop_image(moving, crop_border)

    for dy in range(initial_shift[0] - max_shift, initial_shift[0] + max_shift + 1):
        for dx in range(initial_shift[1] - max_shift, initial_shift[1] + max_shift + 1):
            shifted = shift_and_crop(moving_crop_template, dy, dx)
            min_h = min(base_crop.shape[0], shifted.shape[0])
            min_w = min(base_crop.shape[1], shifted.shape[1])
            loss = ssd_loss(base_crop[:min_h, :min_w], shifted[:min_h, :min_w])
            if loss < best_score:
                best_score = loss
                best_shift = (dy, dx)
    return best_shift


def pyramid_align(base, moving, levels=4, max_shift=15, crop_border=20):
    shift = (0, 0)
    base_l = base.copy()
    moving_l = moving.copy()

    # Build pyramid from coarse to fine.
    for level in range(levels - 1, -1, -1):
        scale = 2 ** level
        if scale > 1:
            b_small = cv2.resize(base, (base.shape[1] // scale, base.shape[0] // scale), interpolation=cv2.INTER_AREA)
            m_small = cv2.resize(moving, (moving.shape[1] // scale, moving.shape[0] // scale), interpolation=cv2.INTER_AREA)
        else:
            b_small = base
            m_small = moving

        shift = align_pair(b_small, m_small, max_shift=max_shift, crop_border=max(crop_border // scale, 2), initial_shift=(shift[0] // 2, shift[1] // 2))
        # Prepare initial shift for next level (finer resolution)
        shift = (shift[0] * 2, shift[1] * 2)

    # Final fine search around refined shift
    shift = align_pair(base, moving, max_shift=4, crop_border=crop_border, initial_shift=shift)
    return shift


def align_channels(blue, green, red, use_pyramid=True):
    if use_pyramid:
        shift_b = pyramid_align(green, blue, levels=5, max_shift=15)
        shift_r = pyramid_align(green, red, levels=5, max_shift=15)
    else:
        shift_b = align_pair(green, blue, max_shift=30)
        shift_r = align_pair(green, red, max_shift=30)

    blue_aligned = shift_and_crop(blue, shift_b[0], shift_b[1])
    red_aligned = shift_and_crop(red, shift_r[0], shift_r[1])
    return blue_aligned, green, red_aligned, shift_b, shift_r


def compose_color(blue, green, red):
    h = min(blue.shape[0], green.shape[0], red.shape[0])
    w = min(blue.shape[1], green.shape[1], red.shape[1])
    return np.dstack((blue[:h, :w], green[:h, :w], red[:h, :w]))


def normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    m = img.min()
    M = img.max()
    if M == m:
        return np.zeros_like(img, dtype=np.uint8)
    out = ((img - m) * 255.0 / (M - m)).clip(0, 255).astype(np.uint8)
    return out


def run(input_path, output_path, show=False):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read grayscale plate
    plate = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if plate is None:
        raise ValueError(f"Cannot read image file: {input_path}")

    try:
        blue, green, red = split_channels(plate)
    except ValueError as e:
        raise

    # Pre-crop to avoid border artifacts
    blue = crop_image(blue, 10)
    green = crop_image(green, 10)
    red = crop_image(red, 10)

    blue_aligned, green_aligned, red_aligned, shift_b, shift_r = align_channels(blue, green, red)

    color = compose_color(blue_aligned, green_aligned, red_aligned)
    color = normalize_to_uint8(color)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, color)

    print(f"Saved aligned RGB image: {output_path}")
    print(f"Estimated shifts (blue vs green): dy={shift_b[0]}, dx={shift_b[1]}")
    print(f"Estimated shifts (red vs green): dy={shift_r[0]}, dx={shift_r[1]}")

    if show:
        cv2.imshow("Aligned RGB", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Lab2: Align Prokudin-Gorskii channels and generate color image")
    parser.add_argument("input", help="Input Prokudin-Gorskii grayscale plate image")
    parser.add_argument("output", nargs="?", default="result_lab2.jpg", help="Output color .jpg path")
    parser.add_argument("--show", action="store_true", help="Show output image window")
    args = parser.parse_args()

    try:
        out = run(args.input, args.output, show=args.show)
        print("Done.")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
