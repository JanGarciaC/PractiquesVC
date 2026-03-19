import os
import numpy as np
import cv2
from PIL import Image

# =========================
# CONFIGURACIÓ
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "fotos", "input")
gt_path = os.path.join(BASE_DIR, "fotos", "groundtruth")
output_path = os.path.join(BASE_DIR, "fotos", "output")

alpha = 0.5
beta = 60
thr = 80

video_output = os.path.join(BASE_DIR, "resultats.avi")

# crear carpetes output si no existeixen
os.makedirs(output_path + "/simple", exist_ok=True)
os.makedirs(output_path + "/gauss", exist_ok=True)
os.makedirs(output_path + "/morph", exist_ok=True)

# FUNCIO PER LLEGIR IMATGES

def read_gray_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)

# TASCA 1 - CARREGAR DATASET

files = sorted(os.listdir(input_path))
files = files[:300]

train_files = files[:150]
test_files = files[150:]

train_images = []

for f in train_files:
    img = read_gray_image(os.path.join(input_path, f))
    train_images.append(img)

train_images = np.stack(train_images)

print("Training images loaded:", train_images.shape)

# TASCA 2 - MITJANA I STD

mu = np.mean(train_images, axis=0)
sigma = np.std(train_images, axis=0)

mu = mu.astype(np.float32)
sigma = sigma.astype(np.float32)

print("Background model computed")

# guardar mean.png
cv2.imwrite(output_path + "/mean.png", mu.astype(np.uint8))

# guardar std.png
cv2.imwrite(output_path + "/std.png", sigma.astype(np.uint8))

# TASCA 3 I 4 - FUNCIONS SEGMENTACIÓ

def simple_subtraction(img):
    diff = np.abs(img - mu)
    fg = diff > thr

    return fg.astype(np.uint8)


def gaussian_model(img):
    diff = np.abs(img - mu)
    fg = diff > (alpha * sigma + beta)

    return fg.astype(np.uint8)


# morfologia millorada
def apply_morphology(mask):

    kernel1 = np.ones((2,2), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)

    # unir parts dels cotxes
    mask = cv2.dilate(mask, kernel1)

    # omplir forats
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    return mask


# TASCA 5 - VIDEO

fourcc = cv2.VideoWriter_fourcc(*'XVID')

height, width = mu.shape

video = cv2.VideoWriter(
    video_output,
    fourcc,
    20,
    (width, height),
    False
)

# TASCA 6 - ACCURACY

def compute_accuracy(pred, gt):

    gt_fg = (gt == 255).astype(np.uint8)

    correct = (pred == gt_fg).sum()
    total = pred.size

    return correct / total

# TEST

acc_simple = []
acc_gauss = []
acc_morph = []

for f in test_files:

    img = read_gray_image(os.path.join(input_path, f))

    gt_name = f.replace("in", "gt")
    gt_name = gt_name.replace("jpg", "png")
    gt = read_gray_image(os.path.join(gt_path, gt_name))

    # model simple
    seg_simple = simple_subtraction(img)

    # model gaussià
    seg_gauss = gaussian_model(img)

    # morfologia
    seg_morph = apply_morphology(seg_gauss)

    # guardar imatge a output
    out_img_simple = (seg_simple * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path + "/simple", f), out_img_simple)

    out_img_gauss = (seg_gauss * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path + "/gauss", f), out_img_gauss)

    out_img_morph = (seg_morph * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out_img_morph)
    min_area = 250
    clean = np.zeros_like(out_img_morph)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean[labels == i] = 255
    out_img_morph = clean
    cv2.imwrite(os.path.join(output_path + "/morph", f), out_img_morph)

    # afegir al video
    video.write(out_img_morph)

    # accuracy
    acc_simple.append(compute_accuracy(out_img_simple, gt))
    acc_gauss.append(compute_accuracy(out_img_gauss, gt))
    acc_morph.append(compute_accuracy(out_img_morph, gt))

video.release()

# RESULTATS

print("\nRESULTATS MITJANS\n")

print("Accuracy model simple:", np.mean(acc_simple))
print("Accuracy model gaussià:", np.mean(acc_gauss))
print("Accuracy amb morfologia:", np.mean(acc_morph))

print("\nVideo guardat a:", video_output)