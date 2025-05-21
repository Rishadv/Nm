
Code for the Healthcare Imaging Analysis :
import cv2
# Load a medical image (grayscale recommended for  X-ray, MRI, etc.)
image = cv2.imread('xray_sample.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image file
# Resize (optional, for display purposes)
image = cv2.resize(image, (512, 512))
# Apply Gaussian Blur to remove noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Display original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Edges Detected', edges)

# Wait until a key is pressed, then close all windows
cv2.waitKey(0)
cv 2.destroyAll
Windows()
