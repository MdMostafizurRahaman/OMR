import cv2
import numpy as np
import os

def test_opencv_bubble_detection():
    """Direct OpenCV bubble detection test"""
    
    # Test images
    test_images = [
        "eng_ans1.jpg",
        "eng_ans2.jpg", 
        "eng_ques.jpg"
    ]
    
    for img_name in test_images:
        img_path = f"d:/OMR/omr/{img_name}"
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing OpenCV detection on: {img_name}")
        print('='*60)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print("Could not load image!")
            continue
            
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        print(f"Image size: {w}x{h}")
        print(f"Mean brightness: {np.mean(gray):.1f}")
        
        # Method 1: HoughCircles detection
        print("\n--- Method 1: HoughCircles ---")
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"Found {len(circles)} circles")
            
            # Draw circles
            for (x, y, r) in circles[:50]:  # Show first 50
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
        else:
            print("No circles detected")
            
        # Method 2: Contour-based detection
        print("\n--- Method 2: Contour Detection ---")
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple thresholding approaches
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_or(thresh1, thresh2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubble_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 800:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.5 <= aspect_ratio <= 2.0:  # Filter by aspect ratio
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.4:  # Reasonably circular
                            bubble_contours.append(contour)
        
        print(f"Found {len(bubble_contours)} bubble contours")
        
        # Draw contours
        for contour in bubble_contours[:100]:  # Show first 100
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
            
        # Method 3: Template matching approach
        print("\n--- Method 3: Grid-based Detection ---")
        
        # Assume answer area is in lower 70% of image
        answer_start_y = int(h * 0.3)
        answer_area = gray[answer_start_y:, :]
        
        # Try to find regular grid patterns
        rows, cols = 25, 4  # 25 questions, 4 options each
        
        if answer_area.shape[0] > 0 and answer_area.shape[1] > 0:
            answer_h, answer_w = answer_area.shape
            
            # Calculate grid spacing
            row_spacing = answer_h // rows
            col_spacing = answer_w // cols
            
            grid_bubbles = 0
            filled_bubbles = []
            
            for row in range(rows):
                for col in range(cols):
                    # Calculate bubble center
                    center_y = answer_start_y + int((row + 0.5) * row_spacing)
                    center_x = int((col + 0.5) * col_spacing)
                    
                    if 0 <= center_x < w and 0 <= center_y < h:
                        # Sample area around center
                        radius = 10
                        x1 = max(0, center_x - radius)
                        x2 = min(w, center_x + radius)
                        y1 = max(0, center_y - radius)
                        y2 = min(h, center_y + radius)
                        
                        roi = gray[y1:y2, x1:x2]
                        if roi.size > 0:
                            avg_intensity = np.mean(roi)
                            
                            # Check if bubble is filled (darker)
                            if avg_intensity < 150:  # Threshold for filled
                                filled_bubbles.append((center_x, center_y, avg_intensity))
                                cv2.circle(image, (center_x, center_y), radius, (0, 255, 255), 2)  # Yellow for filled
                            else:
                                cv2.circle(image, (center_x, center_y), radius//2, (128, 128, 128), 1)  # Gray for empty
                            
                            grid_bubbles += 1
            
            print(f"Grid method: {grid_bubbles} positions checked, {len(filled_bubbles)} potentially filled")
            
            # Show some filled bubble intensities
            if filled_bubbles:
                filled_bubbles.sort(key=lambda x: x[2])  # Sort by intensity
                print("Darkest filled bubbles:")
                for i, (x, y, intensity) in enumerate(filled_bubbles[:10]):
                    print(f"  Bubble at ({x}, {y}): intensity {intensity:.1f}")
        
        # Method 4: Edge-based detection
        print("\n--- Method 4: Edge Detection ---")
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours from edges
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_contours = 0
        for contour in edge_contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                # Fit circle to contour
                if len(contour) >= 5:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if 8 < radius < 20:  # Reasonable bubble size
                        cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 255), 1)  # Magenta
                        circular_contours += 1
        
        print(f"Edge method: found {circular_contours} circular edges")
        


        debug_path = os.path.join(os.path.dirname(__file__), f"debug_opencv_{img_name}")
        cv2.imwrite(debug_path, image)
        debug_img = cv2.imread(debug_path)
    if debug_img is not None:
            h, w = debug_img.shape[:2]
            # Try to detect the answer area using contour/box detection
            gray_dbg = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
            blurred_dbg = cv2.GaussianBlur(gray_dbg, (5, 5), 0)
            edged_dbg = cv2.Canny(blurred_dbg, 50, 150)
            contours_dbg, _ = cv2.findContours(edged_dbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            answer_box = None
            max_area = 0
            for cnt in contours_dbg:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                # More relaxed heuristic: large box, not too high up, aspect ratio reasonable
                if area > max_area and bw > w * 0.4 and bh > h * 0.2 and y > h * 0.1 and bw/bh > 1.5:
                    max_area = area
                    answer_box = (x, y, bw, bh)
            if answer_box is not None:
                x, y, bw, bh = answer_box
                # Draw rectangle and label
                cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 4)
                label = "OMR ans"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                label_x = x + (bw - label_size[0]) // 2
                label_y = max(y - 20, 40)
                cv2.putText(debug_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                # Use this box as answer area for MCQ count
                answer_area = debug_img[y:y+bh, x:x+bw]
                answer_h, answer_w = answer_area.shape[:2]
                rows, cols = 30, 4
                row_spacing = answer_h // rows
                col_spacing = answer_w // cols
                blue_count = 0
                hsv = cv2.cvtColor(answer_area, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([100, 100, 50])
                upper_blue = np.array([140, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                bubble_number = 1
                for row in range(rows):
                    for col in range(cols):
                        center_y = int((row + 0.5) * row_spacing)
                        center_x = int((col + 0.5) * col_spacing)
                        radius = min(row_spacing, col_spacing) // 2
                        y1 = max(0, center_y - radius)
                        y2 = min(answer_h, center_y + radius)
                        x1 = max(0, center_x - radius)
                        x2 = min(answer_w, center_x + radius)
                        roi = mask[y1:y2, x1:x2]
                        if roi.size > 0 and np.sum(roi > 0) > 0.15 * roi.size:
                            blue_count += 1
                            # Draw number at the bubble center on the main debug image
                            abs_cx = x + center_x
                            abs_cy = y + center_y
                            cv2.putText(debug_img, str(bubble_number), (abs_cx-15, abs_cy+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 4, cv2.LINE_AA)
                            bubble_number += 1

                # Annotate count inside the box
                cv2.putText(debug_img, f"MCQ blue: {blue_count}", (x + 30, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)
                cv2.imwrite(debug_path, debug_img)
                print(f"\nDebug image saved: {debug_path}")
                print(f"Blue-marked MCQ bubbles: {blue_count}")

            else:
                # Save the image anyway, with a warning label
                cv2.putText(debug_img, "No answer box detected", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
                cv2.imwrite(debug_path, debug_img)
                print(f"\nDebug image saved: {debug_path}")
                print("Could not detect answer area box for MCQ counting.")
    else:
        print(f"\nDebug image saved: {debug_path}")
        print("Could not reload debug image for blue bubble counting.")

        # Show statistics
        print(f"\nSummary for {img_name}:")
        print(f"  HoughCircles: {len(circles) if circles is not None else 0} circles")
        print(f"  Contours: {len(bubble_contours)} bubble-like shapes")
        print(f"  Grid: {len(filled_bubbles)} potentially filled bubbles")
        print(f"  Edges: {circular_contours} circular edges")

if __name__ == "__main__":
    test_opencv_bubble_detection()