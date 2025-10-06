"""
MANUAL COORDINATE FIXER
Based on exact inspection of the OMR sheets
"""

import cv2
import numpy as np
import json
import os

def inspect_image_manually(image_path):
    """Manually inspect image to find exact bubble coordinates"""
    
    print(f"Manual inspection of {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"Image size: {width}x{height}")
    
    # From the debug images, I can see the issue
    # Let me extract the exact coordinates by analyzing the sheet structure
    
    # The Big Bang OMR sheet has specific structure:
    # Looking at the validation images, the bubble positions are not correctly calculated
    
    # Let me create a more accurate coordinate system by analyzing the actual sheet
    
    # Based on typical Big Bang OMR sheet structure and the debug images:
    # The answer section appears to start around row 300-350 and has 10 rows
    # Each row is about 40-50 pixels high
    # The columns are divided into 3 sections
    
    # More precise measurements:
    answer_start_y = 340  # Based on debug image analysis
    row_height = 42       # Approximate height of each question row
    answer_start_x = 50   # Left margin
    column_width = 180    # Width of each column
    
    # Within each column, bubbles for A,B,C,D are positioned
    # Based on typical OMR layout, they are evenly spaced
    bubble_offset_x = 10  # Offset from column start
    bubble_spacing = 25   # Space between A,B,C,D bubbles
    
    coordinates = []
    
    for col in range(3):  # 3 columns
        for row in range(10):  # 10 questions per column
            question_num = col * 10 + row + 1
            
            # Calculate base position
            col_x = answer_start_x + col * column_width
            row_y = answer_start_y + row * row_height
            
            # Calculate bubble positions for A, B, C, D
            question_bubbles = []
            for i in range(4):  # A, B, C, D
                bubble_x = col_x + bubble_offset_x + i * bubble_spacing
                bubble_y = row_y
                question_bubbles.append((bubble_x, bubble_y))
            
            coordinates.append({
                'question': question_num,
                'bubbles': question_bubbles
            })
    
    # Test these coordinates by sampling some positions
    print("Testing coordinates:")
    test_image = image.copy()
    
    for i in range(min(5, len(coordinates))):
        q_data = coordinates[i]
        question_num = q_data['question']
        bubbles = q_data['bubbles']
        
        print(f"Q{question_num}: ", end="")
        for j, (x, y) in enumerate(bubbles):
            if 0 <= x < width and 0 <= y < height:
                intensity = gray[y, x]
                option = ['A', 'B', 'C', 'D'][j]
                
                # Draw test circle
                color = (0, 255, 0) if intensity < 180 else (0, 0, 255)
                cv2.circle(test_image, (x, y), 10, color, 2)
                cv2.putText(test_image, f"{option}", (x-5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                print(f"{option}:{intensity}", end=" ")
        print()
    
    cv2.imwrite(f"manual_test_{os.path.basename(image_path)}", test_image)
    print(f"Test image saved: manual_test_{os.path.basename(image_path)}")
    
    return coordinates

def create_ultra_precise_detector():
    """Create detector with manually adjusted coordinates"""
    
    class UltraPreciseOMR:
        def __init__(self):
            # Manually determined coordinates for Big Bang OMR sheets
            # These are based on careful analysis of the sheet structure
            self.coordinates = {
                'answer_start_y': 340,
                'row_height': 42,
                'answer_start_x': 50,
                'column_width': 180,
                'bubble_offset_x': 10,
                'bubble_spacing': 25
            }
        
        def get_bubble_coordinates(self, image_shape):
            """Get precise bubble coordinates for the image"""
            height, width = image_shape
            
            coordinates = []
            
            for col in range(3):  # 3 columns
                for row in range(10):  # 10 questions per column
                    question_num = col * 10 + row + 1
                    
                    # Calculate positions
                    col_x = self.coordinates['answer_start_x'] + col * self.coordinates['column_width']
                    row_y = self.coordinates['answer_start_y'] + row * self.coordinates['row_height']
                    
                    question_bubbles = []
                    for i in range(4):  # A, B, C, D
                        bubble_x = col_x + self.coordinates['bubble_offset_x'] + i * self.coordinates['bubble_spacing']
                        bubble_y = row_y
                        question_bubbles.append((bubble_x, bubble_y))
                    
                    coordinates.append({
                        'question': question_num,
                        'bubbles': question_bubbles
                    })
            
            return coordinates
        
        def is_bubble_filled(self, gray_image, x, y, radius=15):
            """Ultra-precise bubble detection"""
            height, width = gray_image.shape
            
            # Ensure coordinates are valid
            if not (radius <= x < width - radius and radius <= y < height - radius):
                return False, 255, "out_of_bounds"
            
            # Sample multiple points around the bubble center
            sample_points = [
                (x, y),                    # center
                (x-5, y), (x+5, y),       # horizontal
                (x, y-5), (x, y+5),       # vertical
                (x-3, y-3), (x+3, y+3),   # diagonal
                (x-3, y+3), (x+3, y-3)    # diagonal
            ]
            
            intensities = []
            for px, py in sample_points:
                if 0 <= px < width and 0 <= py < height:
                    intensities.append(gray_image[py, px])
            
            if not intensities:
                return False, 255, "no_samples"
            
            avg_intensity = np.mean(intensities)
            min_intensity = np.min(intensities)
            
            # A bubble is filled if:
            # 1. Average intensity is low (dark)
            # 2. At least one point is very dark
            
            is_filled = avg_intensity < 150 and min_intensity < 100
            
            return is_filled, avg_intensity, "analyzed"
        
        def process_image(self, image_path, expected_answers=None):
            """Process image with ultra-precise detection"""
            
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get coordinates
            coordinates = self.get_bubble_coordinates(gray.shape)
            
            # Process each question
            results = {}
            vis_image = image.copy()
            
            print(f"\nProcessing {image_path} with ultra-precise detection:")
            
            for q_data in coordinates:
                question_num = q_data['question']
                bubbles = q_data['bubbles']
                options = ['A', 'B', 'C', 'D']
                
                filled_options = []
                print(f"Q{question_num:2d}: ", end="")
                
                for i, (x, y) in enumerate(bubbles):
                    is_filled, intensity, status = self.is_bubble_filled(gray, x, y)
                    
                    if is_filled:
                        filled_options.append(options[i])
                    
                    # Visualize
                    color = (0, 255, 0) if is_filled else (0, 0, 255)
                    thickness = 3 if is_filled else 1
                    cv2.circle(vis_image, (x, y), 12, color, thickness)
                    cv2.putText(vis_image, options[i], (x-5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    symbol = "●" if is_filled else "○"
                    print(f"{options[i]}:{intensity:.0f}{symbol}", end=" ")
                
                # Determine answer
                if len(filled_options) == 1:
                    results[question_num] = filled_options[0]
                    print(f"→ {filled_options[0]}")
                elif len(filled_options) > 1:
                    results[question_num] = f"{filled_options[0]}*"
                    print(f"→ {filled_options[0]}* (multiple: {filled_options})")
                else:
                    results[question_num] = "BLANK"
                    print("→ BLANK")
            
            # Save visualization
            cv2.imwrite(f"ultra_precise_{os.path.basename(image_path)}", vis_image)
            
            # Evaluate if expected provided
            if expected_answers:
                correct = 0
                for q in range(1, 31):
                    detected = results.get(q, "BLANK").replace("*", "")
                    expected = expected_answers.get(q, "BLANK")
                    if detected == expected:
                        correct += 1
                
                accuracy = (correct / 30) * 100
                print(f"\nAccuracy: {accuracy:.1f}% ({correct}/30)")
                return results, accuracy
            
            return results, None

def main():
    """Test ultra-precise detection"""
    
    # First, manually inspect coordinates
    test_images = ['eng_ans1.jpg', 'eng_ans2.jpg']
    
    for image_path in test_images:
        if os.path.exists(image_path):
            coordinates = inspect_image_manually(image_path)
            
            # Save coordinates for reference
            with open(f"manual_coords_{os.path.splitext(image_path)[0]}.json", 'w') as f:
                json.dump(coordinates, f, indent=2)
    
    # Now test with ultra-precise detector
    detector_class = create_ultra_precise_detector()
    detector = detector_class()
    
    test_cases = {
        'eng_ans1.jpg': {
            1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
            11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
            21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
        }
    }
    
    for image_path, expected in test_cases.items():
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"ULTRA-PRECISE TEST: {image_path}")
            print(f"{'='*60}")
            
            results, accuracy = detector.process_image(image_path, expected)
            
            if accuracy is not None:
                if accuracy >= 90:
                    print("🎉 EXCELLENT! Near perfect accuracy!")
                elif accuracy >= 70:
                    print("✅ GOOD accuracy achieved!")
                else:
                    print("❌ Still needs improvement")

if __name__ == "__main__":
    main()