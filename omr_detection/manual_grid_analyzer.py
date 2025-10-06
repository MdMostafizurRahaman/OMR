"""
Manual Grid Analysis Tool
Analyzes the debug images to understand the exact bubble positioning
"""

import cv2
import numpy as np
import json

def analyze_bubble_grid(image_path, debug_image_path):
    """Manually analyze the grid structure from debug images"""
    
    # Load the original image and calibration grid image
    original = cv2.imread(image_path)
    if debug_image_path and os.path.exists(debug_image_path):
        grid_debug = cv2.imread(debug_image_path)
    else:
        grid_debug = original.copy()
    
    print(f"Analyzing {image_path}")
    print(f"Image shape: {original.shape}")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Apply aggressive thresholding to see bubble structure
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Look for circular patterns (bubbles) in the image
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=5, maxRadius=25)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Found {len(circles)} potential bubble circles")
        
        # Draw circles on image for visualization
        circle_image = original.copy()
        for (x, y, r) in circles:
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circle_image, (x, y), 2, (0, 0, 255), 3)
        
        cv2.imwrite(f"circle_analysis_{os.path.basename(image_path)}", circle_image)
        print(f"Circle analysis saved to: circle_analysis_{os.path.basename(image_path)}")
        
        # Group circles by rows and columns
        circles_sorted = sorted(circles, key=lambda c: (c[1], c[0]))  # Sort by y, then x
        
        # Try to identify the grid pattern
        if len(circles) >= 30:  # At least 30 bubbles for questions
            # Group by approximate y-coordinates (rows)
            rows = []
            current_row = []
            last_y = circles_sorted[0][1]
            y_threshold = 30  # Pixels tolerance for same row
            
            for (x, y, r) in circles_sorted:
                if abs(y - last_y) < y_threshold:
                    current_row.append((x, y, r))
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [(x, y, r)]
                    last_y = y
            
            if current_row:
                rows.append(current_row)
            
            print(f"Identified {len(rows)} rows of bubbles")
            for i, row in enumerate(rows[:10]):  # Show first 10 rows
                print(f"Row {i+1}: {len(row)} circles at y≈{row[0][1]}")
        
        # Return circle information for manual calibration
        return circles
    
    return None

def create_manual_calibration(image_path, circles):
    """Create manual calibration based on detected circles"""
    
    if circles is None or len(circles) < 30:
        print("Not enough circles detected for manual calibration")
        return None
    
    # Sort circles by position to identify the grid
    circles_sorted = sorted(circles, key=lambda c: (c[1], c[0]))
    
    # Group into a 10x12 grid (10 rows, 12 columns - 3 columns of 4 bubbles each)
    # This assumes the Big Bang Exam Care layout
    
    calibration = {
        'bubble_positions': [],
        'detected_circles': circles_sorted[:120] if len(circles_sorted) >= 120 else circles_sorted
    }
    
    # Try to organize circles into questions
    # Each question should have 4 consecutive circles (A, B, C, D)
    
    for i in range(0, min(len(circles_sorted), 120), 4):
        if i + 3 < len(circles_sorted):
            question_num = (i // 4) + 1
            question_bubbles = []
            
            for j in range(4):
                x, y, r = circles_sorted[i + j]
                question_bubbles.append((x, y))
            
            calibration['bubble_positions'].append({
                'question': question_num,
                'bubbles': question_bubbles
            })
    
    return calibration

def main():
    """Main function to analyze grid structure"""
    
    images_to_analyze = ['eng_ans1.jpg', 'eng_ans2.jpg']
    
    for image_path in images_to_analyze:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"ANALYZING: {image_path}")
            print(f"{'='*50}")
            
            # Analyze bubble structure
            circles = analyze_bubble_grid(image_path, None)
            
            if circles is not None:
                # Create manual calibration
                manual_cal = create_manual_calibration(image_path, circles)
                
                if manual_cal:
                    # Save manual calibration
                    cal_file = f"manual_calibration_{os.path.splitext(image_path)[0]}.json"
                    with open(cal_file, 'w') as f:
                        json.dump(manual_cal, f, indent=2, default=str)
                    print(f"Manual calibration saved to: {cal_file}")
                    
                    print(f"Detected {len(manual_cal['bubble_positions'])} questions")
                    
                    # Show first few questions for verification
                    print("\\nFirst 5 questions:")
                    for q_data in manual_cal['bubble_positions'][:5]:
                        q_num = q_data['question']
                        bubbles = q_data['bubbles']
                        print(f"Q{q_num}: {bubbles}")

if __name__ == "__main__":
    import os
    main()