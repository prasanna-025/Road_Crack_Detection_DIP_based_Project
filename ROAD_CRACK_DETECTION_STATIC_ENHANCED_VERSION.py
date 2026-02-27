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

    # Apply Non-Local Means Denoising (Preserves small cracks)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(denoised)

    # Apply Gaussian Blur to reduce small noise
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Gabor Filtering for crack texture enhancement
    kernel_size = 31
    theta = np.pi / 4
    sigma = 5
    lamda = 10
    gamma = 0.5
    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(clahe_enhanced, cv2.CV_8UC3, gabor_kernel)

    # Apply Morphological Closing (Dilation -> Erosion) to enhance cracks
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply Otsu’s Thresholding
    _, otsu_threshold = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Adaptive Thresholding (Better for uneven lighting)
    adaptive_thresh = cv2.adaptiveThreshold(clahe_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Top-Hat & Bottom-Hat Transform (Highlight cracks)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top_hat = cv2.morphologyEx(clahe_enhanced, cv2.MORPH_TOPHAT, structuring_element)  # Highlights cracks
    bottom_hat = cv2.morphologyEx(clahe_enhanced, cv2.MORPH_BLACKHAT, structuring_element)  # Removes large shadows

    # Convert to Lab Color Space and extract L-channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    # Extract crack edges from the original image
    crack_overlay = cv2.bitwise_and(image, image, mask=closed)

    return (image, gray, hist, denoised, clahe_enhanced, blurred, edges, gabor_filtered, closed, otsu_threshold, 
            adaptive_thresh, top_hat, bottom_hat, l_channel, crack_overlay)

# Process each image in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    (original, gray, hist, denoised, clahe_enhanced, blurred, edges, gabor_filtered, closed, otsu_threshold, 
     adaptive_thresh, top_hat, bottom_hat, l_channel, crack_overlay) = detect_cracks(image_path)

    # Display results
    plt.figure(figsize=(18, 12))

    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(4, 4, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")

    plt.subplot(4, 4, 3)
    plt.plot(hist)
    plt.title("Histogram of Grayscale Image")

    plt.subplot(4, 4, 4)
    plt.imshow(denoised, cmap="gray")
    plt.title("Denoised Image (Non-Local Means)")

    plt.subplot(4, 4, 5)
    plt.imshow(clahe_enhanced, cmap="gray")
    plt.title("CLAHE Enhanced Image")

    plt.subplot(4, 4, 6)
    plt.imshow(blurred, cmap="gray")
    plt.title("Gaussian Blurred Image")

    plt.subplot(4, 4, 7)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")

    plt.subplot(4, 4, 8)
    plt.imshow(gabor_filtered, cmap="gray")
    plt.title("Gabor Filtered Image")

    plt.subplot(4, 4, 9)
    plt.imshow(closed, cmap="gray")
    plt.title("Morphological Closing")

    plt.subplot(4, 4, 10)
    plt.imshow(otsu_threshold, cmap="gray")
    plt.title("Otsu’s Thresholding")

    plt.subplot(4, 4, 11)
    plt.imshow(adaptive_thresh, cmap="gray")
    plt.title("Adaptive Thresholding")

    plt.subplot(4, 4, 12)
    plt.imshow(top_hat, cmap="gray")
    plt.title("Top-Hat Transform (Highlight Cracks)")

    plt.subplot(4, 4, 13)
    plt.imshow(bottom_hat, cmap="gray")
    plt.title("Bottom-Hat Transform (Remove Shadows)")

    plt.subplot(4, 4, 14)
    plt.imshow(l_channel, cmap="gray")
    plt.title("L-Channel (Lab Color Space)")

    plt.subplot(4, 4, 15)
    plt.imshow(cv2.cvtColor(crack_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Crack Overlay on Original Image")

    plt.tight_layout()
    plt.show()