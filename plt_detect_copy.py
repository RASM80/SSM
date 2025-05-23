import cv2
import numpy as np
import pytesseract
import sys

# Set the path to the Tesseract executable (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python plate_recognition.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Could not load image")
    sys.exit(1)

# Show original image
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

# Show detected plate
cv2.imshow('Detected Plate', plate)
cv2.waitKey(0)

# Detect and crop the blue section on the left
hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 100]  # Filter small contours
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort by x-coordinate
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    plate = plate[:, x + w:]  # Crop out the blue section
    print(f"Cropped blue section: x={x}, w={w}")
else:
    print("No blue section detected")
cv2.imshow('Cropped Plate', plate)
cv2.waitKey(0)


# Preprocess the plate image
plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

# --- Adjustable Parameters ---
# Thresholding method (choose one by uncommenting)
threshold_method = 'otsu'  # Options: 'otsu', 'adaptive', 'manual'

# For manual thresholding
manual_threshold_value = 150  # Adjust between 0-255

# For adaptive thresholding
adaptive_block_size = 5  # Must be odd, adjust for character size
adaptive_constant = 3    # Adjust for contrast

# Morphological operations
use_morphology = False    # Set to True to enable
kernel_size = (3, 3)      # Adjust kernel size
morph_iterations = 1      # Number of dilation/erosion iterations

# Resizing
resize_factor = 2.0       # Increase (e.g., 2.0) if characters are too small

# Contour detection
contour_retrieval_mode = cv2.RETR_LIST  # Options: RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE
min_char_height = 0.1     # Min height as fraction of plate height
max_char_height = 0.9     # Max height as fraction of plate height
min_char_width = 0.05     # Min width as fraction of plate width
max_char_width = 0.3      # Max width as fraction of plate width

# --- Thresholding ---
if threshold_method == 'otsu':
    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
elif threshold_method == 'adaptive':
    plate_thresh = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, adaptive_constant)
elif threshold_method == 'manual':
    _, plate_thresh = cv2.threshold(plate_gray, manual_threshold_value, 255, cv2.THRESH_BINARY)
else:
    print("Invalid threshold method")
    sys.exit(1)

# Optional: Invert the threshold (if characters are black on white)
plate_thresh = cv2.bitwise_not(plate_thresh)

# --- Morphological Operations ---
if use_morphology:
    kernel = np.ones(kernel_size, np.uint8)
    plate_thresh = cv2.dilate(plate_thresh, kernel, iterations=morph_iterations)
    plate_thresh = cv2.erode(plate_thresh, kernel, iterations=morph_iterations)

# --- Resizing ---
if resize_factor != 1.0:
    plate_thresh = cv2.resize(plate_thresh, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

# Get the dimensions of plate_thresh
height, width = plate_thresh.shape

# Calculate rectangle dimensions
rec_width = int(0.218 * width)
rec_height = int(0.264 * height)

# Define the top-left and bottom-right points of the rectangle
pt1 = (width - rec_width, 0)
pt2 = (width, rec_height)

# Draw the filled black rectangle
cv2.rectangle(plate_thresh, pt1, pt2, 0, -1)

# Show thresholded plate
cv2.imshow('Thresholded Plate', plate_thresh)
cv2.waitKey(0)

# Detect character edges using contours
char_contours, _ = cv2.findContours(plate_thresh, contour_retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

# Get plate dimensions
height, width = plate_thresh.shape

# Filter contours to likely characters based on size
char_candidates = []
for contour in char_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (min_char_height * height < h < max_char_height * height) and (min_char_width * width < w < max_char_width * width):
        char_candidates.append((x, contour))

# Sort contours by x-coordinate (left to right)
char_candidates.sort(key=lambda c: c[0])

# Visualize detected contours
plate_with_contours = cv2.cvtColor(plate_thresh, cv2.COLOR_GRAY2BGR)
for i, (x, contour) in enumerate(char_candidates):
    cv2.drawContours(plate_with_contours, [contour], -1, (0, 255, 0), 2)
    cv2.putText(plate_with_contours, str(i+1), (x, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.imshow('Detected Contours', plate_with_contours)
cv2.waitKey(0)

# Check number of detected contours
print(f"Detected {len(char_candidates)} contours")
if len(char_candidates) < 7:
    print("Detected fewer than 8 contours. Try adjusting thresholding, morphology, or contour parameters.")
    sys.exit(1)
elif len(char_candidates) > 8:
    char_candidates = char_candidates[:8]  # Take first 8

# Extract character images
char_images = []
for i, (_, contour) in enumerate(char_candidates):
    x, y, w, h = cv2.boundingRect(contour)
    char_img = plate_thresh[y:y+h, x:x+w]
    char_img_with_border = cv2.copyMakeBorder(char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    char_images.append(char_img_with_border)
    display_img = cv2.resize(char_img_with_border, (100, 100), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(f'Character {i+1}', display_img)
    cv2.waitKey(0)

# Define whitelists for OCR
num_whitelist = '٠١٢٣٤٥٦٧٨٩'
letter_whitelist = 'بجدسصطقلمنوهی'

# Perform OCR
result = ''
for i, char_img in enumerate(char_images):
    if i == 2:  # Assuming position 3 is a letter
        custom_config = f'--psm 10 -c tessedit_char_whitelist={letter_whitelist}'
    else:  # Numbers
        custom_config = f'--psm 10 -c tessedit_char_whitelist={num_whitelist}'
    char = pytesseract.image_to_string(char_img, lang='fas', config=custom_config)
    char = char.strip()
    result += char[0] if char else '?'

# Close windows
cv2.destroyAllWindows()

# Print result
print(f"Detected plate: {result}")