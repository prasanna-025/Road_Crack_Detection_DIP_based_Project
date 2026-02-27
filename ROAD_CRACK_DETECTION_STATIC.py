import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the folder containing road images
folder_path = r"C:\Users\HomeLaptop\OneDrive\Desktop\SRIDHAR\THIRD YEAR\SEM6\DIP\PROJECT\road_crack_images"

def detect_cracks(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Histogram (for visualization)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Gabor Filtering
    kernel_size = 31  # Kernel size for better feature extraction
    theta = np.pi / 4  # Orientation
    sigma = 5  # Standard deviation
    lamda = 10  # Wavelength
    gamma = 0.5  # Aspect ratio

    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(clahe_enhanced, cv2.CV_8UC3, gabor_kernel)

    # Apply Morphological Closing (Dilation -> Erosion) to enhance cracks
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply Otsu’s Thresholding
    _, otsu_threshold = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract crack edges from the original image
    crack_overlay = cv2.bitwise_and(image, image, mask=closed)

    return image, gray, hist, clahe_enhanced, blurred, edges, gabor_filtered, closed, otsu_threshold, crack_overlay

# Process each image in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    original, gray, hist, clahe_enhanced, blurred, edges, gabor_filtered, closed, otsu_threshold, crack_overlay = detect_cracks(image_path)

    # Display results
    plt.figure(figsize=(18, 9))

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")

    plt.subplot(3, 3, 3)
    plt.plot(hist)
    plt.title("Histogram of Grayscale Image")

    plt.subplot(3, 3, 4)
    plt.imshow(clahe_enhanced, cmap="gray")
    plt.title("CLAHE Enhanced Image")

    plt.subplot(3, 3, 5)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")

    plt.subplot(3, 3, 6)
    plt.imshow(gabor_filtered, cmap="gray")
    plt.title("Gabor Filtering")

    plt.subplot(3, 3, 7)
    plt.imshow(closed, cmap="gray")
    plt.title("Morphological Closing (Crack Enhancement)")

    plt.subplot(3, 3, 8)
    plt.imshow(otsu_threshold, cmap="gray")
    plt.title("Otsu’s Thresholding")

    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(crack_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Crack Overlay on Original Image")

    plt.tight_layout()
    plt.show()