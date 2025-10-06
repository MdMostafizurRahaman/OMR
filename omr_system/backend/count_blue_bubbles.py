import cv2
import numpy as np
import os

def count_blue_bubbles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return 0
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define blue color range in HSV
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter small contours (noise)
    bubble_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]
    return len(bubble_contours)

if __name__ == "__main__":
    # Use absolute paths for images
    base_dir = r"d:/OMR/omr_system/backend"
    debug_images = [
        "debug_opencv_eng_ans1.jpg",
        "debug_opencv_eng_ans2.jpg",
        "debug_opencv_eng_ques.jpg"
    ]
    for img_name in debug_images:
        img_path = os.path.join(base_dir, img_name)
        count = count_blue_bubbles(img_path)
        print(f"{img_path}: {count} blue-marked bubbles detected")
