import cv2
import numpy as np
from typing import Dict, List, Tuple
import json

class SmartOMRDetector:
    """
    Advanced OMR detector that learns from actual Bengali OMR patterns
    """
    
    def __init__(self):
        self.debug_mode = True
        
    def detect_omr_structure(self, image):
        """
        Analyze the actual OMR structure to understand layout - IMPROVED FOR DARK OMR
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        height, width = gray.shape
        
        # Check if this is an inverted OMR
        avg_intensity = np.mean(gray)
        is_inverted = avg_intensity < 127
        
        print(f"Image type: {'Inverted' if is_inverted else 'Normal'} (avg intensity: {avg_intensity:.1f})")
        
        # Apply appropriate preprocessing based on OMR type
        if is_inverted:
            # For dark background OMR, we need different preprocessing
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use different thresholding for inverted OMR
            # We want to detect both filled (very dark) and empty (white) bubbles
            _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
            
            # Also create an inverted binary to catch filled bubbles
            _, binary_inv = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
            
            # Combine both to detect all circular shapes
            combined = cv2.bitwise_or(binary, binary_inv)
            
        else:
            # Original preprocessing for normal OMR
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, combined = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for bubble-like shapes with adjusted criteria
        potential_bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Adjust area range based on OMR type
            if is_inverted:
                min_area, max_area = 30, 600  # Slightly different range for inverted
            else:
                min_area, max_area = 50, 800  # Original range
                
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.5 <= aspect_ratio <= 2.0:  # More lenient aspect ratio
                    # Check if it's in the answer area (skip header)
                    if y > height * 0.25:  # Below header
                        
                        # Additional circularity check
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:  # More lenient circularity
                                potential_bubbles.append({
                                    'x': x + w//2,
                                    'y': y + h//2,
                                    'w': w,
                                    'h': h,
                                    'area': area,
                                    'circularity': circularity
                                })
        
        print(f"Found {len(potential_bubbles)} potential bubbles")
        
        # Need at least 200 bubbles for 100 questions (4 options each)
        if len(potential_bubbles) < 150:  
            print("Not enough bubbles detected - trying alternative detection")
            return self.alternative_bubble_detection(gray, is_inverted)
            
        # Step 2: Group bubbles into grid structure
        return self.analyze_grid_structure(potential_bubbles, width, height)
    
    def alternative_bubble_detection(self, gray, is_inverted):
        """
        Alternative bubble detection method for difficult cases
        """
        height, width = gray.shape
        
        # Try edge detection approach
        edges = cv2.Canny(gray, 50, 150)
        
        # Find circles using HoughCircles
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
        
        potential_bubbles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Only consider circles in answer area
                if y > height * 0.25:
                    potential_bubbles.append({
                        'x': x,
                        'y': y,
                        'w': r*2,
                        'h': r*2,
                        'area': np.pi * r * r,
                        'circularity': 1.0  # Perfect circle from Hough detection
                    })
        
        print(f"Alternative detection found {len(potential_bubbles)} circles")
        
        if len(potential_bubbles) >= 100:
            return self.analyze_grid_structure(potential_bubbles, width, height)
        
        return None
    
    def analyze_grid_structure(self, bubbles, width, height):
        """
        Analyze bubble positions to understand the grid structure
        """
        # Sort by Y position to group into rows
        bubbles.sort(key=lambda b: b['y'])
        
        # Group into rows based on Y similarity
        rows = []
        current_row = []
        
        if len(bubbles) > 0:
            last_y = bubbles[0]['y']
            row_threshold = 25  # pixels
            
            for bubble in bubbles:
                if abs(bubble['y'] - last_y) > row_threshold:
                    if len(current_row) >= 3:  # Valid row should have multiple bubbles
                        rows.append(sorted(current_row, key=lambda b: b['x']))
                    current_row = [bubble]
                    last_y = bubble['y']
                else:
                    current_row.append(bubble)
            
            if len(current_row) >= 3:
                rows.append(sorted(current_row, key=lambda b: b['x']))
        
        print(f"Detected {len(rows)} rows")
        
        # Step 3: Analyze column structure
        if len(rows) >= 20:  # Should have around 25 rows per column
            return self.map_questions_to_bubbles(rows, width)
        
        return None
    
    def map_questions_to_bubbles(self, rows, width):
        """
        Map detected bubble rows to actual questions
        """
        # Bengali OMR typically has 4 columns with 25 questions each
        # Let's determine column boundaries by analyzing X positions
        
        all_x_positions = []
        for row in rows:
            for bubble in row:
                all_x_positions.append(bubble['x'])
        
        all_x_positions.sort()
        
        # Find natural breaks in X positions to determine columns
        column_breaks = []
        for i in range(1, len(all_x_positions)):
            if all_x_positions[i] - all_x_positions[i-1] > 100:  # Significant gap
                column_breaks.append((all_x_positions[i-1] + all_x_positions[i]) // 2)
        
        # Should have 3 breaks for 4 columns
        if len(column_breaks) < 2:
            # Fallback: divide width into 4 equal parts
            column_breaks = [width//4, width//2, 3*width//4]
        
        column_boundaries = [0] + column_breaks + [width]
        
        print(f"Column boundaries: {column_boundaries}")
        
        # Now map each row to questions
        question_map = {}
        
        for row_idx, row in enumerate(rows):
            if row_idx >= 25:  # Maximum 25 rows per column
                break
                
            for col_idx in range(min(4, len(column_boundaries)-1)):
                question_num = (col_idx * 25) + row_idx + 1
                
                if question_num > 100:
                    break
                
                # Find bubbles in this column for this row
                col_start = column_boundaries[col_idx]
                col_end = column_boundaries[col_idx + 1]
                
                col_bubbles = [b for b in row if col_start <= b['x'] < col_end]
                
                if len(col_bubbles) >= 4:  # Should have A, B, C, D
                    # Sort by X position for A, B, C, D order
                    col_bubbles.sort(key=lambda b: b['x'])
                    question_map[question_num] = col_bubbles[:4]  # Take first 4
        
        print(f"Mapped {len(question_map)} questions")
        return question_map
    
    def extract_answers(self, image, question_map):
        """
        Extract answers using the detected structure - OPTIMIZED FOR DARK/INVERTED OMR
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check if this is an inverted OMR (dark background, white bubbles)
        avg_intensity = np.mean(gray)
        is_inverted = avg_intensity < 127  # If average is dark, it's inverted
        
        print(f"Average image intensity: {avg_intensity:.1f}")
        print(f"OMR type: {'Inverted (dark background)' if is_inverted else 'Normal (light background)'}")
        
        answers = {}
        
        for question_num, bubbles in question_map.items():
            best_option = None
            best_score = -1
            
            bubble_scores = []
            
            for option_idx, bubble in enumerate(bubbles):
                option_letter = chr(65 + option_idx)  # A, B, C, D
                
                # Extract region around bubble
                x_start = max(0, bubble['x'] - bubble['w']//2)
                x_end = min(gray.shape[1], bubble['x'] + bubble['w']//2)
                y_start = max(0, bubble['y'] - bubble['h']//2)
                y_end = min(gray.shape[0], bubble['y'] + bubble['h']//2)
                
                roi = gray[y_start:y_end, x_start:x_end]
                
                if roi.size > 0:
                    if is_inverted:
                        # For inverted OMR: filled bubbles are DARKER (black) on dark background
                        # Empty bubbles are white circles
                        avg_intensity = np.mean(roi)
                        min_intensity = np.min(roi)
                        max_intensity = np.max(roi)
                        
                        # For filled bubbles in inverted OMR, we want VERY dark regions
                        very_dark_pixels = np.sum(roi < 50) / roi.size  # Very dark pixels
                        dark_pixels = np.sum(roi < 100) / roi.size      # Dark pixels
                        
                        # Standard deviation (filled bubbles should be uniformly dark)
                        intensity_std = np.std(roi)
                        
                        # For inverted OMR: lower average = more filled
                        darkness_score = (255 - avg_intensity) / 255  # Normalized darkness
                        uniformity_score = max(0, (100 - intensity_std) / 100)  # More uniform = better
                        
                        # Combined score (higher = more likely filled)
                        fill_score = (darkness_score * 0.4 + very_dark_pixels * 0.3 + 
                                    dark_pixels * 0.2 + uniformity_score * 0.1) * 100
                        
                        print(f"Q{question_num}{option_letter}: avg={avg_intensity:.1f}, "
                              f"dark_ratio={dark_pixels:.2f}, std={intensity_std:.1f}, "
                              f"score={fill_score:.1f}")
                        
                    else:
                        # Original logic for normal OMR (light background, dark filled bubbles)
                        avg_intensity = np.mean(roi)
                        min_intensity = np.min(roi)
                        dark_pixel_ratio = np.sum(roi < 100) / roi.size
                        
                        # Combined score for normal OMR (higher = darker = more filled)
                        fill_score = (255 - avg_intensity) + (255 - min_intensity) + (dark_pixel_ratio * 100)
                        
                        print(f"Q{question_num}{option_letter}: avg={avg_intensity:.1f}, "
                              f"dark_ratio={dark_pixel_ratio:.2f}, score={fill_score:.1f}")
                    
                    bubble_scores.append({
                        'option': option_letter,
                        'score': fill_score,
                        'avg_intensity': avg_intensity
                    })
            
            # Find the best scoring bubble
            if bubble_scores:
                # Sort by score (descending for both normal and inverted)
                bubble_scores.sort(key=lambda x: x['score'], reverse=True)
                
                best_bubble = bubble_scores[0]
                second_best = bubble_scores[1] if len(bubble_scores) > 1 else None
                
                # Check if there's a clear winner
                score_difference = 0
                if second_best:
                    score_difference = best_bubble['score'] - second_best['score']
                
                # Adaptive threshold based on OMR type
                if is_inverted:
                    min_score_threshold = 30  # Lower threshold for inverted OMR
                    min_difference_threshold = 10
                else:
                    min_score_threshold = 50  # Higher threshold for normal OMR
                    min_difference_threshold = 15
                
                if (best_bubble['score'] > min_score_threshold and 
                    score_difference > min_difference_threshold):
                    answers[str(question_num)] = best_bubble['option']
                    print(f"✓ Q{question_num}: {best_bubble['option']} "
                          f"(score: {best_bubble['score']:.1f}, diff: {score_difference:.1f})")
                else:
                    print(f"? Q{question_num}: No clear answer "
                          f"(best: {best_bubble['score']:.1f}, diff: {score_difference:.1f})")
        
        return answers
    
    def process_omr(self, image_path):
        """
        Main processing function
        """
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        print(f"Processing: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Step 1: Detect OMR structure
        question_map = self.detect_omr_structure(image)
        
        if question_map is None:
            print("Failed to detect OMR structure")
            return {}
        
        # Step 2: Extract answers
        answers = self.extract_answers(image, question_map)
        
        print(f"Extracted {len(answers)} answers")
        return answers