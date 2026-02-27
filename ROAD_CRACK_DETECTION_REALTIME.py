import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the IP camera URL and the folder path
ip_camera_url = "http://100.65.243.238:4747/video"  # Replace with your actual IP camera stream URL
folder_path = r"C:\Users\HomeLaptop\OneDrive\Desktop\DIP_PROJECT"  # Replace with your actual folder path

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    hist_eq = cv2.equalizeHist(gray)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours of the cracks
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Overlay detected edges on the original image
    overlay = frame.copy()

    # Draw contours and calculate crack lengths
    for i, contour in enumerate(contours):
        # Calculate the length of the contour (crack)
        length = cv2.arcLength(contour, closed=False)

        # Draw the contour on the overlay
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)  # Red color, thickness 2

        # Get the centroid of the contour to place the text
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Display the length of the crack near the contour
        cv2.putText(overlay, f"L: {length:.2f} px", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green color, thickness 2

    return gray, hist_eq, edges, overlay

def add_label(image, label):
    # Add a label to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2
    margin = 10  # Margin from the top-left corner

    cv2.putText(image, label, (margin, margin + 30), font, font_scale, font_color, thickness)
    return image

def plot_histogram(gray):
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Plot histogram using matplotlib
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black')
    plt.xlim([0, 256])

    # Convert matplotlib plot to an image
    plt.gcf().canvas.draw()
    img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

    # Close the plot to free memory
    plt.close()

    return img

def live_crack_detection():
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("Error: Could not open IP camera stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray, hist_eq, edges, overlay = process_frame(frame)

        # Convert grayscale images to 3-channel format for stacking
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hist_eq = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Add labels to each image
        frame = add_label(frame, "Original")
        gray = add_label(gray, "Grayscale")
        hist_eq = add_label(hist_eq, "Histogram Equalization")
        edges = add_label(edges, "Canny Edges")
        overlay = add_label(overlay, "Overlay (Cracks in Red)")

        # Generate histogram for the grayscale image
        histogram = plot_histogram(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))

        # Resize images to ensure they have the same dimensions
        h, w, _ = frame.shape
        gray = cv2.resize(gray, (w, h))
        hist_eq = cv2.resize(hist_eq, (w, h))
        edges = cv2.resize(edges, (w, h))
        overlay = cv2.resize(overlay, (w, h))
        histogram = cv2.resize(histogram, (w, h))

        # Stack images horizontally (3 images per row)
        top_row = np.hstack([frame, gray, hist_eq])
        bottom_row = np.hstack([edges, overlay, histogram])
        
        # Stack the two rows vertically
        combined_frame = np.vstack([top_row, bottom_row])

        # Show the combined image
        cv2.imshow("Live Crack Detection (All in One Frame)", combined_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
     
