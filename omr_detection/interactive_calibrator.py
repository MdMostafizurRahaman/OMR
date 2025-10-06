"""
Interactive OMR Calibration and Testing Tool
Helps fine-tune bubble detection parameters for 100% accuracy
"""

import cv2
import numpy as np
import json
import os
from precise_omr_detector import PreciseOMRDetector

class InteractiveCalibrator:
    def __init__(self):
        self.detector = PreciseOMRDetector()
        self.current_image = None
        self.calibration = None
        
    def analyze_image_characteristics(self, image_path):
        """Analyze image characteristics to understand bubble filling"""
        print(f"\nAnalyzing image: {image_path}")
        
        # Load and process image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing methods
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Get calibration
        calibration = self.detector.manual_calibration(image_path)
        
        print(f"Image shape: {image.shape}")
        print(f"Grid bounds: {calibration['grid_bounds']}")
        
        # Sample some bubble positions for analysis
        sample_positions = []
        for i in range(0, min(10, len(calibration['bubble_positions']))):
            q_data = calibration['bubble_positions'][i]
            sample_positions.extend([(q_data['question'], j, pos) for j, pos in enumerate(q_data['bubbles'])])
        
        print(f"\nSampling bubble characteristics:")
        print(f"{'Question':<8} {'Option':<6} {'Position':<15} {'Intensity':<10} {'Fill%':<8} {'Status'}")
        print("-" * 70)
        
        for q_num, opt_idx, (x, y) in sample_positions[:20]:
            result = self.detector.detect_bubbles_in_region(enhanced, x, y, radius=15)
            option_letter = ['A', 'B', 'C', 'D'][opt_idx]
            status = "FILLED" if result['is_filled'] else "EMPTY"
            
            print(f"Q{q_num:<7} {option_letter:<6} ({x},{y}){'':<5} {result['avg_intensity']:<10.1f} {result['fill_percentage']:<8.3f} {status}")
    
    def find_optimal_parameters(self, image_path, known_answers):
        """Find optimal detection parameters by testing different thresholds"""
        print(f"\nFinding optimal parameters for {image_path}")
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        calibration = self.detector.manual_calibration(image_path)
        
        # Test different threshold values
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        radii = [10, 12, 15, 18, 20]
        
        best_score = 0
        best_params = None
        
        print(f"\nTesting parameter combinations:")
        print(f"{'Threshold':<10} {'Radius':<8} {'Correct':<8} {'Wrong':<6} {'Blank':<6} {'Score':<8}")
        print("-" * 50)
        
        for threshold in thresholds:
            for radius in radii:
                self.detector.bubble_threshold = threshold
                
                # Test detection with these parameters
                detected_answers = {}
                
                for q_data in calibration['bubble_positions']:
                    question_num = q_data['question']
                    bubble_positions = q_data['bubbles']
                    
                    question_results = []
                    option_labels = ['A', 'B', 'C', 'D']
                    
                    for i, (bubble_x, bubble_y) in enumerate(bubble_positions):
                        bubble_result = self.detector.detect_bubbles_in_region(
                            enhanced, bubble_x, bubble_y, radius=radius
                        )
                        bubble_result['option'] = option_labels[i]
                        question_results.append(bubble_result)
                    
                    # Determine the selected answer
                    filled_bubbles = [r for r in question_results if r['is_filled']]
                    
                    if len(filled_bubbles) == 1:
                        detected_answers[question_num] = filled_bubbles[0]['option']
                    elif len(filled_bubbles) > 1:
                        most_filled = max(filled_bubbles, key=lambda x: x['detection_confidence'])
                        detected_answers[question_num] = most_filled['option']
                    else:
                        detected_answers[question_num] = "BLANK"
                
                # Evaluate against known answers
                correct = 0
                wrong = 0
                blank = 0
                
                for q in range(1, 31):
                    detected = detected_answers.get(q, "BLANK")
                    known = known_answers.get(q, "BLANK")
                    
                    if detected == "BLANK" and known == "BLANK":
                        correct += 1
                    elif detected == known:
                        correct += 1
                    elif detected == "BLANK":
                        blank += 1
                    else:
                        wrong += 1
                
                score = (correct / 30) * 100
                
                print(f"{threshold:<10.1f} {radius:<8} {correct:<8} {wrong:<6} {blank:<6} {score:<8.1f}%")
                
                if score > best_score:
                    best_score = score
                    best_params = (threshold, radius, detected_answers)
        
        if best_params:
            print(f"\nBest parameters: Threshold={best_params[0]}, Radius={best_params[1]}")
            print(f"Best score: {best_score:.1f}%")
            
            # Show detailed results for best parameters
            print(f"\nDetailed results with best parameters:")
            detected = best_params[2]
            known = known_answers
            
            print(f"{'Q':<3} {'Detected':<9} {'Expected':<9} {'Status'}")
            print("-" * 35)
            
            for q in range(1, 31):
                det = detected.get(q, "BLANK")
                exp = known.get(q, "BLANK")
                status = "✓" if det == exp else "✗"
                print(f"{q:<3} {det:<9} {exp:<9} {status}")
        
        return best_params

def main():
    """Interactive calibration and testing"""
    calibrator = InteractiveCalibrator()
    
    # Test images and their known answers
    test_cases = {
        'eng_ans1.jpg': {
            1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
            11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
            21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
        },
        'eng_ans2.jpg': {
            1: 'D', 2: 'C', 3: 'A', 4: 'D', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
            11: 'A', 12: 'D', 13: 'B', 14: 'C', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'BLANK',
            21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'BLANK', 28: 'BLANK', 29: 'BLANK', 30: 'BLANK'
        }
    }
    
    for image_path, known_answers in test_cases.items():
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"PROCESSING: {image_path}")
            print(f"{'='*60}")
            
            # Analyze image characteristics
            calibrator.analyze_image_characteristics(image_path)
            
            # Find optimal parameters
            best_params = calibrator.find_optimal_parameters(image_path, known_answers)
            
            if best_params:
                # Save optimal parameters
                optimal_config = {
                    'image': image_path,
                    'threshold': best_params[0],
                    'radius': best_params[1],
                    'score': (sum(1 for q in range(1,31) if best_params[2].get(q) == known_answers.get(q)) / 30) * 100
                }
                
                config_file = f"optimal_config_{os.path.splitext(image_path)[0]}.json"
                with open(config_file, 'w') as f:
                    json.dump(optimal_config, f, indent=2)
                
                print(f"Optimal configuration saved to: {config_file}")

if __name__ == "__main__":
    main()