import cv2
import numpy as np
import pytesseract
import sys

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python plate_recognition_updated.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Could not load image")
    sys.exit(1)

# Show the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(0)

# Canny edge detection
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Initialize plate variable
plate = None

# Loop through contours to find the plate
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 2 < aspect_ratio < 5 and area > 1000:  # Adjust thresholds as needed
            plate = image[y:y+h, x:x+w]
            break

if plate is None:
    print("No plate detected")
    sys.exit(1)

# Show the detected plate
cv2.imshow('Detected Plate', plate)
cv2.waitKey(0)

# Preprocess the plate image
plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
cv2.imshow('Plate Grayscale', plate_gray)
cv2.waitKey(0)

_, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Thresholded Plate', plate_thresh)
cv2.waitKey(0)

# Perform OCR
text = pytesseract.image_to_string(plate_thresh, lang='ara', config='--psm 7')

# Clean the text
text = ' '.join(text.split())

if text:
    print(f"Detected text: {text}")
else:
    print("No text detected")

# Close all windows
cv2.destroyAllWindows()