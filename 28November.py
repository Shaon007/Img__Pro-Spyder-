from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny

# Load the image
image_path = "Stone- (10).jpg"  # Replace with your image file path
image = Image.open(image_path)

# Define crop boxes for right and left kidneys
right_kidney_box = (130, 220, 230, 300)  # Adjust based on your needs
left_kidney_box = (300, 230, 390, 310)  # Adjust based on your needs

# Crop the right and left kidneys
right_kidney = image.crop(right_kidney_box)
left_kidney = image.crop(left_kidney_box)

# Convert the cropped images to grayscale (if not already)
right_kidney_gray = right_kidney.convert('L')
left_kidney_gray = left_kidney.convert('L')

# Convert the grayscale images to NumPy arrays
right_kidney_array = np.array(right_kidney_gray)
left_kidney_array = np.array(left_kidney_gray)

# Apply normal Canny edge detection (without sigma)
right_kidney_edge_normal = canny(right_kidney_array)
left_kidney_edge_normal = canny(left_kidney_array)

# Apply Canny edge detection with sigma (for bright edges like kidney stones)
right_kidney_edge = canny(right_kidney_array, sigma=7)
left_kidney_edge = canny(left_kidney_array, sigma=7)

# Display both normal and sigma-enhanced Canny edge-detected images
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Normal right kidney edge display
axes[0, 0].imshow(right_kidney_edge_normal, cmap='gray')
axes[0, 0].set_title("Right Kidney - Normal Canny")
axes[0, 0].axis("off")

# Sigma right kidney edge display
axes[0, 1].imshow(right_kidney_edge, cmap='gray')
axes[0, 1].set_title("Right Kidney -Stone with sigma 7")
axes[0, 1].axis("off")

# Normal left kidney edge display
axes[1, 0].imshow(left_kidney_edge_normal, cmap='gray')
axes[1, 0].set_title("Left Kidney - Normal Canny")
axes[1, 0].axis("off")

# Sigma left kidney edge display
axes[1, 1].imshow(left_kidney_edge, cmap='gray')
axes[1, 1].set_title("Left Kidney - Canny with Sigma 7")
axes[1, 1].axis("off")

plt.show()
