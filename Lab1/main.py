import os
import numpy as np
from PIL import Image
import cv2

train_path = 'fotos/train'

image_files = [f for f in os.listdir(train_path) if f.endswith('.jpg')]

first_image = Image.open(os.path.join(train_path, image_files[0])).convert('L')
width, height = first_image.size

sum_pixels = np.zeros((height, width), dtype=np.float64)
sum_sq_pixels = np.zeros((height, width), dtype=np.float64)
num_images = len(image_files)

for img_file in image_files:
    img_path = os.path.join(train_path, img_file)
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    sum_pixels += img_array
    sum_sq_pixels += img_array ** 2

mean = sum_pixels / num_images

variance = (sum_sq_pixels / num_images) - (mean ** 2)
std_dev = np.sqrt(variance)

mean_image = Image.fromarray(np.uint8(mean))
mean_image.save('mean.png')

std_image = Image.fromarray(np.uint8(std_dev))
std_image.save('std.png')

thr = 50  

for img_file in image_files:
    img_path = os.path.join(train_path, img_file)
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    diff = img_array - mean
    binary = np.where(np.abs(diff) <= thr, 0, 1)
    binary_image = Image.fromarray((binary * 255).astype(np.uint8))
    output_path = os.path.join('fotos/processed', img_file.replace('.jpg', '_binary.png'))
    binary_image.save(output_path)

alpha = 1.4  # Parámetro de scaling de la desviación estándar
beta = 12  # Parámetro de offset

for img_file in image_files:
    img_path = os.path.join(train_path, img_file)
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    diff = img_array - mean
    threshold = alpha * std_dev + beta
    binary = np.where(np.abs(diff) > threshold, 1, 0)
    
    # Aplicar morfología para eliminar ruido
    binary_np = (binary * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_np, cv2.MORPH_OPEN, kernel)
    
    binary_image = Image.fromarray(cleaned)
    output_path = os.path.join('fotos/processed', img_file.replace('.jpg', '_binary_alphabeta.png'))
    binary_image.save(output_path)

print("Processed images saved in fotos/processed/")
print(f"Parámetros: α={alpha}, β={beta}")

# Crear vídeo amb els resultats
processed_path = 'fotos/processed'
processed_files = sorted([f for f in os.listdir(processed_path) if f.endswith('_binary.png')])

    # Llegir la primera imatge per obtenir dimensions
first_img_path = os.path.join(processed_path, processed_files[0])
first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
height, width = first_img.shape
    
    # Crear el vídeo amb alta qualitat per evitar pixelació
video_path = 'resultats_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec amb millor qualitat
fps = 24  # Frames per second
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)
    
for file in processed_files:
    img_path = os.path.join(processed_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    video.write(img)
    
video.release()

