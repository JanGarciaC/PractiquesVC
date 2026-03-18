import os
import cv2
import numpy as np
import time

# Carpetes d'entrada i sortida
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "fotos", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "fotos", "output")


# -------------------------
# FUNCIONS BÀSIQUES
# -------------------------

def crop_borders(img, percent=0.1):
    h, w = img.shape  # obtenir alçada i amplada

    # calcular quant retallar a cada costat
    crop_h = int(h * percent)
    crop_w = int(w * percent)

    # retornar només la part central (sense vores)
    return img[crop_h:h - crop_h, crop_w:w - crop_w]


def split_channels(img):
    # dividir la imatge en 3 parts iguals verticalment
    h = img.shape[0] // 3

    # cada tros correspon a un canal
    b = img[0:h]          # primer terç
    g = img[h:2*h]        # segon terç
    r = img[2*h:3*h]      # tercer terç

    return b, g, r


def shift_image(img, dx, dy):
    h, w = img.shape

    # crear una imatge buida del mateix tamany
    result = np.zeros_like(img)

    # calcular la zona de destinació dins la imatge final
    x_start = max(0, dx)
    x_end = min(h, h + dx)

    y_start = max(0, dy)
    y_end = min(w, w + dy)

    # copiar els píxels desplaçats
    result[x_start:x_end, y_start:y_end] = \
        img[x_start - dx:x_end - dx, y_start - dy:y_end - dy]

    return result


# -------------------------
# COMPARACIÓ D'IMATGES
# -------------------------

def ncc(a, b):
    # convertir a float per evitar problemes de tipus
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # restar la mitjana → centrar les dades
    a = a - np.mean(a)
    b = b - np.mean(b)

    # calcular el denominador (normalització)
    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))

    # evitar divisió per zero
    if denom == 0:
        return -1

    # retornem la correlació
    return np.sum(a*b) / denom


# -------------------------
# CERCA DEL DESPLAÇAMENT
# -------------------------

def find_shift(ref, target, max_shift=15):

    best_score = -1  # millor puntuació trobada
    best_shift = (0, 0)

    # retallar bordes per evitar soroll
    ref_crop = crop_borders(ref)

    # provar totes les combinacions de desplaçament
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):

            # desplaçar la imatge objectiu
            shifted = shift_image(target, dx, dy)

            # retallar també la imatge desplaçada
            shifted_crop = crop_borders(shifted)

            # comparar amb NCC
            score = ncc(ref_crop, shifted_crop)

            # guardar el millor resultat
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    return best_shift


def pyramid_align(ref, target, levels=4):

    # cas base: imatge petita → cerca directa
    if levels == 0 or min(ref.shape) < 64:
        return find_shift(ref, target, 15)

    # reduir resolució (imatge més petita)
    ref_small = cv2.pyrDown(ref)
    target_small = cv2.pyrDown(target)

    # calcular desplaçament a escala petita
    dx, dy = pyramid_align(ref_small, target_small, levels - 1)

    # escalar el resultat (per tornar a mida original)
    dx *= 2
    dy *= 2

    best_shift = (dx, dy)
    best_score = -1

    ref_crop = crop_borders(ref)

    # refinar al voltant del resultat trobat
    for i in range(-2, 3):
        for j in range(-2, 3):

            sx = dx + i
            sy = dy + j

            shifted = shift_image(target, sx, sy)
            shifted_crop = crop_borders(shifted)

            score = ncc(ref_crop, shifted_crop)

            if score > best_score:
                best_score = score
                best_shift = (sx, sy)

    return best_shift


# -------------------------
# ALINEACIÓ I PROCESSAMENT
# -------------------------

def align_channels(b, g, r):

    # fem servir el canal verd com a referència
    ref = g

    # calcular desplaçaments
    shift_b = pyramid_align(ref, b)
    shift_r = pyramid_align(ref, r)

    # aplicar desplaçaments
    b_aligned = shift_image(b, *shift_b)
    r_aligned = shift_image(r, *shift_r)

    # ajustar mida comuna (evitar diferències)
    h = min(b_aligned.shape[0], g.shape[0], r_aligned.shape[0])
    w = min(b_aligned.shape[1], g.shape[1], r_aligned.shape[1])

    return b_aligned[:h, :w], g[:h, :w], r_aligned[:h, :w]


def crop_valid_region(b, g, r):

    # crear màscara de píxels vàlids (no negres)
    mask = (b > 0) & (g > 0) & (r > 0)

    coords = np.column_stack(np.where(mask))

    # si no hi ha píxels vàlids → no retallar
    if coords.size == 0:
        return b, g, r

    # trobar límits mínims i màxims
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # retallar la regió vàlida
    return b[y0:y1, x0:x1], g[y0:y1, x0:x1], r[y0:y1, x0:x1]


def enhance(img):

    # normalitzar valors entre 0 i 255
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img_norm.astype(np.uint8)


# -------------------------
# PIPELINE
# -------------------------

def process_image(path):

    start = time.time()

    # llegir imatge en escala de grisos
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, 0

    # separar canals
    b, g, r = split_channels(img)

    # alinear canals
    b, g, r = align_channels(b, g, r)

    # eliminar zones negres
    b, g, r = crop_valid_region(b, g, r)

    # retall final per netejar vores
    percent = 0.05
    b = crop_borders(b, percent)
    g = crop_borders(g, percent)
    r = crop_borders(r, percent)

    # combinar canals en una imatge RGB
    color = np.dstack((b, g, r))

    # calcular temps
    elapsed = time.time() - start

    # millorar contrast
    color = enhance(color)

    return color, elapsed


def save_image(img, input_path):

    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)

    output_path = os.path.join(OUTPUT_DIR, name + "_color.jpg")

    cv2.imwrite(output_path, img)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(INPUT_DIR):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(INPUT_DIR, file)

            print(f"Processant: {file}")

            img, t = process_image(path)

            if img is not None:
                save_image(img, path)
                print(f"Temps: {t:.3f} s")