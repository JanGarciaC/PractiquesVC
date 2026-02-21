######################################################################
### VC i PSIV                                                      ###
### Lab 0 (basat en material de Gemma Rotger)                      ###
######################################################################


# Hello! Welcome to the computer vision LAB. 
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


## PROBLEM 1 (+0.5) --------------------------------------------------
#  READ THE CAMERAMAN IMAGE. 
print('Reading the image...')
imatge = cv2.imread('img/cameraman.jpg', cv2.IMREAD_GRAYSCALE)


## PROBLEM 2 (+0.5) --------------------------------------------------
#  SHOW THE CAMERAMAN IMAGE
plt.figure(1)
plt.imshow(imatge, 'gray')
plt.title('Cameraman image')
plt.show()

## PROBLEM 3 (+2.0) --------------------------------------------------
#  Negative efect using a double for instruction
print("Computing the negative effect using a double for instruction...")
t=time.time()

rows, cols = imatge.shape
imatge_neg = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        imatge_neg[i, j] = 255 - imatge[i, j]

elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

plt.figure(1)
plt.imshow(imatge_neg, 'gray')
plt.title('Negative effect using double for loop')
plt.show()

# Negative efect using a vectorial instruction
print("Computing the negative effect using a vectorial instruction...")
t=time.time()

imatge_neg_vector = 255 - imatge

elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

plt.figure(2)
plt.title('Negative effect using vectorial instruction')
plt.imshow(imatge_neg_vector, 'gray')
plt.show()

# You sould see that results in figures 1 and 2 are the same but times
# are much different.

## PROBLEM 4 (+2.0) --------------------------------------------------
# Give some color (red, green or blue)

print("Creating colored image - Method 1: Creating empty image and filling channels...")

# Method 1: Creating an empty image and filling each channel
r = imatge
g = imatge_neg_vector
b = imatge

im_col_method1 = np.zeros((imatge.shape[0], imatge.shape[1], 3), dtype=np.uint8)
im_col_method1[:,:,0] = r  # Red channel
im_col_method1[:,:,1] = g  # Green channel
im_col_method1[:,:,2] = b  # Blue channel

plt.figure(1)
plt.imshow(im_col_method1)
plt.title('Method 1: Empty image + channel assignment')
plt.show()

print("Creating colored image - Method 2: Using np.dstack...")

# Method 2: Using np.dstack
im_col_method2 = np.dstack((r, g, b))

plt.figure(2)
plt.imshow(im_col_method2)
plt.title('Method 2: Using np.dstack')
plt.show()

print("Both methods produce the same result!")


## PROBLEM 5 (+1.0) --------------------------------------------------

print("Writing images to disk...")
# Write the original image
cv2.imwrite('img/cameraman_original.jpg', imatge)
# Write the negative image
cv2.imwrite('img/cameraman_negative.jpg', imatge_neg_vector)
# Write the colored image
cv2.imwrite('img/cameraman_colored.jpg', im_col_method1)
print('Images written successfully')

## PROBLEM 6 (+1.0) --------------------------------------------------
print("Extracting lines from the image...")
# Extract line 128 from the grayscale image

lin128 = imatge[128, :]
plt.figure(1)
plt.plot(lin128)

# Add a horizontal line with the mean value
mean_val = np.mean(lin128)
plt.axhline(y=mean_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
plt.legend([f'Mean={mean_val:.2f}'])
plt.title('Line 128 from grayscale image')
plt.show()

# Extract line 128 from the colored image
lin128rgb = im_col_method1[128, :, :]
plt.figure(2)
plt.plot(lin128rgb[:, 0], 'r', label='Red')
plt.plot(lin128rgb[:, 1], 'g', label='Green')
plt.plot(lin128rgb[:, 2], 'b', label='Blue')

# Add horizontal lines with the mean values for each channel
mean_red = np.mean(lin128rgb[:, 0])
mean_green = np.mean(lin128rgb[:, 1])
mean_blue = np.mean(lin128rgb[:, 2])
plt.axhline(y=mean_red, color='r', linestyle='--', linewidth=1, alpha=0.7)
plt.axhline(y=mean_green, color='g', linestyle='--', linewidth=1, alpha=0.7)
plt.axhline(y=mean_blue, color='b', linestyle='--', linewidth=1, alpha=0.7)

plt.legend([f'Red (mean={mean_red:.2f})', f'Green (mean={mean_green:.2f})', f'Blue (mean={mean_blue:.2f})'])
plt.title('Line 128 from colored image')
plt.show()

## PROBLEM 7 (+2) ----------------------------------------------------
print("Computing histogram using np.histogram...")
# Compute the histogram.

t=time.time()
hist, bins = np.histogram(imatge, bins=256, range=(0, 256))
elapsed=time.time()-t

print('Elapsed time is '+str(elapsed)+' seconds')

plt.figure(1)
plt.plot(hist)
plt.title('Histogram using np.histogram')
plt.show()

print("Computing histogram using manual loop...")

t=time.time()

h = np.zeros(256, dtype=int)
for i in range(imatge.shape[0]):
    for j in range(imatge.shape[1]):
        h[imatge[i, j]] += 1

elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

plt.figure(2)
plt.plot(h)
plt.title('Histogram using manual loop')
plt.show()

## PROBLEM 8 Binarize the image text.png (+1) ------------------------

print("Reading Alice text image...")
# Read the image (convert to grayscale if necessary)
imtext = cv2.imread('img/alice.jpg', cv2.IMREAD_GRAYSCALE)

# Show the original image
plt.figure(1)
plt.imshow(imtext, 'gray')
plt.title('Original Alice image')
plt.show()

# Calculate and show the histogram
print("Computing histogram of the text image...")
hist_text, bins_text = np.histogram(imtext, bins=256, range=(0, 256))

th1 = 120   # Low threshold
th2 = 175   # Optimal threshold (adjust visually based on histogram)
th3 = 230   # High threshold

plt.figure(2)
plt.plot(hist_text)
plt.xlabel('Gray level')
plt.ylabel('Frequency')
plt.title('Histogram of text image')
# Mark potential threshold values
plt.axvline(x=th1, color='r', linestyle='--', alpha=0.5, label=f'Low threshold ({th1})')
plt.axvline(x=th2, color='g', linestyle='--', alpha=0.5, label=f'Optimal threshold ({th2})')
plt.axvline(x=th3, color='b', linestyle='--', alpha=0.5, label=f'High threshold ({th3})')
plt.legend()
plt.show()

print(f"Applying thresholds: {th1} (low), {th2} (optimal), {th3} (high)")

# Apply the thresholds as binary images (0/1 or False/True)
# Below threshold = 0 (black), Above threshold = 1 (white)
threshimtext1 = (imtext > th1).astype(np.uint8)
threshimtext2 = (imtext > th2).astype(np.uint8)
threshimtext3 = (imtext > th3).astype(np.uint8)

# Show the original image and the segmentations in a subplot
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# First row: show original image and labels
ax[0, 0].remove()
ax[0, 1].imshow(imtext, 'gray')
ax[0, 1].set_title('Original image')
ax[0, 1].axis('off')
ax[0, 2].remove()

# Second row: show the three binary threshold results
ax[1, 0].imshow(threshimtext1, 'gray')
ax[1, 0].set_title(f'Threshold = {th1} (Low - Underestimated)')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshimtext2, 'gray')
ax[1, 1].set_title(f'Threshold = {th2} (Optimal)')
ax[1, 1].axis('off')

ax[1, 2].imshow(threshimtext3, 'gray')
ax[1, 2].set_title(f'Threshold = {th3} (High - Overestimated)')
ax[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("Binary segmentation complete!")


## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab0_NIU.zip 
# (put matlab file lab0.m and python file lab0.py in the same zip file)
# Example lab0_1234567.zip