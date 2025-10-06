"""
Precise Bubble Detector for Big Bang Exam Care OMR Sheets
Based on exact analysis of the sheet structure
"""

import cv2
import numpy as np
import json
import os

def analyze_bigbang_sheet_structure(image_path):
    """
    Analyze the specific structure of Big Bang Exam Care OMR sheets
    """
    print(f"Analyzing Big Bang Exam Care sheet: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"Image dimensions: {width}x{height}")
    
    # Based on the Big Bang sheet structure:
    # - The answer section starts after "Omr ans" text
    # - There are 3 columns of questions  
    # - Each column has 10 questions (1-10, 11-20, 21-30)
    # - Each question has 4 options (A, B, C, D)
    
    # From visual inspection of the sheets, the answer grid appears to be:
    # - Located in roughly the bottom 60% of the image
    # - The grid spans most of the width
    # - Questions are arranged in 3 columns
    
    # More precise measurements based on the sheet structure
    answer_region_start_y = int(height * 0.40)  # Answer region starts at 40% down
    answer_region_height = int(height * 0.55)   # Takes up 55% of height
    answer_region_start_x = int(width * 0.08)   # Small margin from left
    answer_region_width = int(width * 0.84)     # Use 84% of width
    
    # Extract the answer region
    answer_region = gray[answer_region_start_y:answer_region_start_y + answer_region_height,
                        answer_region_start_x:answer_region_start_x + answer_region_width]
    
    # Save the extracted region for analysis
    cv2.imwrite(f"answer_region_{os.path.basename(image_path)}", answer_region)
    
    # Calculate precise bubble positions based on grid structure
    calibration = calculate_precise_bubble_positions(
        answer_region_start_x, answer_region_start_y, 
        answer_region_width, answer_region_height
    )
    
    # Validate the calibration by checking some sample positions
    validate_bubble_positions(gray, calibration, image_path)
    
    return calibration

def calculate_precise_bubble_positions(start_x, start_y, width, height):
    """
    Calculate precise bubble positions for Big Bang Exam Care sheets
    """
    # Grid structure: 3 columns, 10 rows, 4 options per question
    num_columns = 3
    num_rows = 10
    options_per_question = 4
    
    # Calculate spacing
    column_width = width / num_columns
    row_height = height / num_rows
    
    # Bubble positioning within each cell
    # Based on typical OMR sheet layouts, bubbles are positioned:
    # - Vertically centered in each row
    # - Horizontally distributed across the options area
    
    bubble_positions = []
    
    for col in range(num_columns):
        for row in range(num_rows):
            question_num = col * 10 + row + 1
            
            # Base position for this question
            cell_left = start_x + col * column_width
            cell_top = start_y + row * row_height
            cell_center_y = cell_top + row_height / 2
            
            # Calculate bubble positions for A, B, C, D
            # Assume bubbles are evenly distributed in the left portion of each cell
            options_area_width = column_width * 0.5  # Use left 50% of cell for options
            option_spacing = options_area_width / options_per_question
            
            question_bubbles = []
            for option in range(options_per_question):
                bubble_x = int(cell_left + option_spacing * (option + 0.5))
                bubble_y = int(cell_center_y)
                question_bubbles.append((bubble_x, bubble_y))
            
            bubble_positions.append({
                'question': question_num,
                'bubbles': question_bubbles
            })
    
    calibration = {
        'grid_bounds': (start_x, start_y, width, height),
        'bubble_positions': bubble_positions,
        'column_width': column_width,
        'row_height': row_height
    }
    
    return calibration

def validate_bubble_positions(image, calibration, image_path):
    """
    Validate bubble positions by checking intensity at detected positions
    """
    print(f"Validating bubble positions for {image_path}")
    
    # Create visualization image
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Check first 5 questions as samples
    for i, q_data in enumerate(calibration['bubble_positions'][:5]):
        question_num = q_data['question']
        bubbles = q_data['bubbles']
        
        print(f"Q{question_num}:", end=" ")
        for j, (x, y) in enumerate(bubbles):
            # Check the intensity at this position
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                intensity = image[y, x]
                option_letter = ['A', 'B', 'C', 'D'][j]
                
                # Draw the bubble position
                color = (0, 255, 0) if intensity < 150 else (255, 0, 0)  # Green if dark (filled), red if light
                cv2.circle(vis_image, (x, y), 8, color, 2)
                cv2.putText(vis_image, f"{option_letter}", (x-5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                status = "FILLED" if intensity < 150 else "EMPTY"
                print(f"{option_letter}:{intensity}({status})", end=" ")
        print()
    
    # Save visualization
    cv2.imwrite(f"bubble_validation_{os.path.basename(image_path)}", vis_image)
    print(f"Validation image saved: bubble_validation_{os.path.basename(image_path)}")

def create_enhanced_omr_detector():
    """
    Create an enhanced OMR detector with corrected calibration
    """
    
    class EnhancedOMRDetector:
        def __init__(self):
            self.bubble_threshold = 0.3  # Lower threshold for better detection
            self.debug = True
        
        def detect_answers(self, image_path, answer_key=None):
            """Detect answers using the enhanced calibration"""
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get calibration
            calibration = analyze_bigbang_sheet_structure(image_path)
            
            # Detect answers
            answers = {}
            bubble_analysis = []
            
            for q_data in calibration['bubble_positions']:
                question_num = q_data['question']
                bubble_positions = q_data['bubbles']
                
                question_results = []
                option_labels = ['A', 'B', 'C', 'D']
                
                for i, (bubble_x, bubble_y) in enumerate(bubble_positions):
                    # Enhanced bubble detection
                    if 0 <= bubble_x < gray.shape[1] and 0 <= bubble_y < gray.shape[0]:
                        # Sample a small area around the bubble center
                        sample_size = 10
                        y1 = max(0, bubble_y - sample_size)
                        y2 = min(gray.shape[0], bubble_y + sample_size)
                        x1 = max(0, bubble_x - sample_size)
                        x2 = min(gray.shape[1], bubble_x + sample_size)
                        
                        bubble_area = gray[y1:y2, x1:x2]
                        
                        if bubble_area.size > 0:
                            avg_intensity = np.mean(bubble_area)
                            min_intensity = np.min(bubble_area)
                            
                            # A bubble is filled if the average intensity is low
                            # or if there are very dark pixels in the area
                            is_filled = avg_intensity < 180 or min_intensity < 100
                            
                            bubble_result = {
                                'option': option_labels[i],
                                'question': question_num,
                                'center': (bubble_x, bubble_y),
                                'avg_intensity': avg_intensity,
                                'min_intensity': min_intensity,
                                'is_filled': is_filled,
                                'confidence': (200 - avg_intensity) / 200 if avg_intensity < 200 else 0
                            }
                            
                            question_results.append(bubble_result)
                            bubble_analysis.append(bubble_result)
                
                # Determine the selected answer
                filled_bubbles = [r for r in question_results if r['is_filled']]
                
                if len(filled_bubbles) == 1:
                    answers[question_num] = filled_bubbles[0]['option']
                elif len(filled_bubbles) > 1:
                    # Multiple answers - select the one with highest confidence
                    most_confident = max(filled_bubbles, key=lambda x: x['confidence'])
                    answers[question_num] = f"{most_confident['option']}*"  # Mark as multiple
                else:
                    answers[question_num] = "BLANK"
            
            # Evaluate against answer key if provided
            evaluation = None
            if answer_key:
                evaluation = self.evaluate_answers(answers, answer_key)
            
            return {
                'answers': answers,
                'bubble_analysis': bubble_analysis,
                'calibration': calibration,
                'evaluation': evaluation
            }
        
        def evaluate_answers(self, detected_answers, answer_key):
            """Evaluate detected answers against answer key"""
            correct = 0
            wrong = 0
            blank = 0
            multiple = 0
            
            for q in range(1, 31):
                detected = detected_answers.get(q, "BLANK")
                expected = answer_key.get(q, "BLANK")
                
                if detected == "BLANK" and expected == "BLANK":
                    correct += 1
                elif "*" in str(detected):  # Multiple answers
                    multiple += 1
                elif detected == expected:
                    correct += 1
                elif detected == "BLANK":
                    blank += 1
                else:
                    wrong += 1
            
            return {
                'summary': {
                    'correct': correct,
                    'wrong': wrong,
                    'blank': blank,
                    'multiple': multiple,
                    'total': 30,
                    'score': (correct / 30) * 100
                }
            }
    
    return EnhancedOMRDetector()

def main():
    """Test the enhanced OMR detector"""
    
    # Test images and their known answers
    test_cases = {
        'eng_ans1.jpg': {
            1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
            11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
            21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
        }
    }
    
    detector = create_enhanced_omr_detector()
    
    for image_path, answer_key in test_cases.items():
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"TESTING: {image_path}")
            print(f"{'='*60}")
            
            results = detector.detect_answers(image_path, answer_key)
            
            # Show results
            answers = results['answers']
            print(f"\nDetected Answers:")
            for q in range(1, 31):
                answer = answers.get(q, 'BLANK')
                expected = answer_key.get(q, 'BLANK')
                status = "✓" if answer == expected else "✗"
                print(f"Q{q:2d}: {answer:<6} (expected: {expected}) {status}")
            
            if results['evaluation']:
                eval_summary = results['evaluation']['summary']
                print(f"\nEvaluation:")
                print(f"Correct: {eval_summary['correct']}")
                print(f"Wrong: {eval_summary['wrong']}")
                print(f"Blank: {eval_summary['blank']}")
                print(f"Multiple: {eval_summary['multiple']}")
                print(f"Score: {eval_summary['score']:.1f}%")

if __name__ == "__main__":
    main()