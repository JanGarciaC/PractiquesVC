import os
import numpy as np
from PIL import Image

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

print("Processed images saved in fotos/processed/")