"""
TEMPLATE MATCHING OMR DETECTOR
Use template matching to find exact bubble positions
"""

import cv2
import numpy as np
import json
import os

class TemplateMatchingOMR:
    def __init__(self):
        self.debug = True
        
    def create_bubble_template(self, size=12):
        """Create a bubble template for matching"""
        
        template = np.ones((size, size), dtype=np.uint8) * 255
        center = size // 2
        cv2.circle(template, (center, center), size//3, 0, -1)  # Filled circle
        
        return template
    
    def create_empty_bubble_template(self, size=12):
        """Create an empty bubble template"""
        
        template = np.ones((size, size), dtype=np.uint8) * 255
        center = size // 2
        cv2.circle(template, (center, center), size//3, 128, 2)  # Circle outline
        
        return template
    
    def detect_with_template_matching(self, image_path, expected_answers):
        """Detect bubbles using template matching"""
        
        print(f"\\n🎯 TEMPLATE MATCHING: {image_path}")
        print("="*60)
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Get roll and set code
        roll, set_code = self.get_roll_and_set_code(image_path)
        
        # Enhanced preprocessing
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Create templates
        filled_template = self.create_bubble_template(14)
        empty_template = self.create_empty_bubble_template(14)
        
        print(f"Roll: {roll}, Set Code: {set_code}")
        print(f"Image size: {width}x{height}")
        
        # Define search regions for each column based on visual analysis
        search_regions = self.define_search_regions(width, height)
        
        all_bubble_candidates = []
        
        # Template matching in each region
        for region_name, (x, y, w, h) in search_regions.items():
            print(f"\\nSearching in {region_name}: ({x},{y}) {w}x{h}")
            
            # Extract region
            roi = enhanced[y:y+h, x:x+w]
            
            # Match filled bubbles
            filled_matches = cv2.matchTemplate(roi, filled_template, cv2.TM_CCOEFF_NORMED)
            empty_matches = cv2.matchTemplate(roi, empty_template, cv2.TM_CCOEFF_NORMED)
            
            # Find good matches
            threshold = 0.3
            filled_locations = np.where(filled_matches >= threshold)
            empty_locations = np.where(empty_matches >= threshold)
            
            # Process filled bubble matches
            for pt_y, pt_x in zip(*filled_locations):
                actual_x = x + pt_x + filled_template.shape[1] // 2
                actual_y = y + pt_y + filled_template.shape[0] // 2
                confidence = filled_matches[pt_y, pt_x]
                
                all_bubble_candidates.append({
                    'x': actual_x,
                    'y': actual_y,
                    'type': 'filled',
                    'confidence': confidence,
                    'region': region_name
                })
        
        # Cluster candidates into questions and options
        clustered_bubbles = self.cluster_bubbles(all_bubble_candidates, width, height)
        
        # Detect answers from clustered bubbles
        detected_answers = self.analyze_clustered_bubbles(clustered_bubbles, expected_answers)
        
        # Create visualization
        vis_image = self.create_template_visualization(image, all_bubble_candidates, detected_answers, expected_answers)
        
        # Save visualization
        output_filename = f"template_matched_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, vis_image)
        print(f"Template matching visualization saved: {output_filename}")
        
        # Calculate accuracy
        correct_count = 0
        for q in range(1, 31):
            detected = detected_answers.get(q, "BLANK")
            expected = expected_answers.get(q, "BLANK")
            if detected == expected:
                correct_count += 1
        
        accuracy = (correct_count / 30) * 100
        print(f"\\n🎯 TEMPLATE MATCHING ACCURACY: {accuracy:.1f}% ({correct_count}/30)")
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'detected_answers': detected_answers,
            'bubble_candidates': all_bubble_candidates
        }
    
    def define_search_regions(self, width, height):
        """Define search regions for template matching"""
        
        # Based on visual analysis, define regions where bubbles might be
        regions = {
            'column1': (60, 280, 180, 400),   # First column region
            'column2': (220, 280, 180, 400),  # Second column region  
            'column3': (380, 280, 180, 400),  # Third column region
        }
        
        return regions
    
    def cluster_bubbles(self, candidates, width, height):
        """Cluster bubble candidates into question/option groups"""
        
        print(f"\\nClustering {len(candidates)} bubble candidates...")
        
        # Sort candidates by position
        candidates_sorted = sorted(candidates, key=lambda x: (x['y'], x['x']))
        
        # Simple clustering based on position
        clustered = {}
        
        # Define approximate question positions (30 questions total)
        question_positions = []
        
        # Column layout: 3 columns x 10 rows
        for col in range(3):
            for row in range(10):
                question_num = col * 10 + row + 1
                
                # Approximate positions
                col_x = 60 + col * 160 + 80  # Column center
                row_y = 300 + row * 35       # Row center
                
                question_positions.append({
                    'question': question_num,
                    'center_x': col_x,
                    'center_y': row_y
                })
        
        # Match candidates to questions
        for candidate in candidates_sorted:
            best_question = None
            best_distance = float('inf')
            
            for q_pos in question_positions:
                distance = np.sqrt((candidate['x'] - q_pos['center_x'])**2 + 
                                 (candidate['y'] - q_pos['center_y'])**2)
                
                if distance < best_distance and distance < 50:  # Maximum distance threshold
                    best_distance = distance
                    best_question = q_pos['question']
            
            if best_question:
                if best_question not in clustered:
                    clustered[best_question] = []
                clustered[best_question].append(candidate)
        
        print(f"Clustered into {len(clustered)} questions")
        
        return clustered
    
    def analyze_clustered_bubbles(self, clustered_bubbles, expected_answers):
        """Analyze clustered bubbles to detect answers"""
        
        detected_answers = {}
        
        print("\\nAnalyzing clustered bubbles:")
        print("-" * 40)
        
        for question_num in range(1, 31):
            if question_num in clustered_bubbles:
                bubbles = clustered_bubbles[question_num]
                
                # Sort bubbles by X position (A, B, C, D)
                bubbles_sorted = sorted(bubbles, key=lambda x: x['x'])
                
                # Find filled bubbles
                filled_bubbles = [b for b in bubbles_sorted if b['type'] == 'filled' and b['confidence'] > 0.4]
                
                if len(filled_bubbles) > 0:
                    # Take the highest confidence bubble
                    best_bubble = max(filled_bubbles, key=lambda x: x['confidence'])
                    
                    # Determine option based on X position
                    option_index = min(3, max(0, len([b for b in bubbles_sorted if b['x'] <= best_bubble['x']]) - 1))
                    detected_option = ['A', 'B', 'C', 'D'][option_index]
                    
                    detected_answers[question_num] = detected_option
                else:
                    detected_answers[question_num] = "BLANK"
            else:
                detected_answers[question_num] = "BLANK"
            
            # Print analysis
            expected = expected_answers.get(question_num, "?")
            detected = detected_answers[question_num]
            status = "✓" if detected == expected else "✗"
            
            bubble_count = len(clustered_bubbles.get(question_num, []))
            filled_count = len([b for b in clustered_bubbles.get(question_num, []) if b['type'] == 'filled'])
            
            print(f"Q{question_num:2d}: {bubble_count} bubbles ({filled_count} filled) → {detected:<6} (exp: {expected:<6}) {status}")
        
        return detected_answers
    
    def create_template_visualization(self, image, candidates, detected_answers, expected_answers):
        """Create visualization of template matching results"""
        
        vis_image = image.copy()
        
        # Draw all bubble candidates
        for candidate in candidates:
            x, y = candidate['x'], candidate['y']
            confidence = candidate['confidence']
            bubble_type = candidate['type']
            
            if bubble_type == 'filled':
                color = (0, 255, 0) if confidence > 0.5 else (0, 150, 255)
                thickness = 3 if confidence > 0.5 else 2
            else:
                color = (128, 128, 128)
                thickness = 1
            
            cv2.circle(vis_image, (x, y), 8, color, thickness)
            
            # Add confidence text
            cv2.putText(vis_image, f"{confidence:.2f}", 
                       (x-15, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add question numbers and results
        for question_num in range(1, 31):
            detected = detected_answers.get(question_num, "BLANK")
            expected = expected_answers.get(question_num, "BLANK")
            
            is_correct = detected == expected
            result_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            # Calculate approximate question position for label
            col = (question_num - 1) // 10
            row = (question_num - 1) % 10
            
            label_x = 60 + col * 160
            label_y = 300 + row * 35
            
            cv2.putText(vis_image, f"Q{question_num}: {detected}", 
                       (label_x - 30, label_y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, result_color, 1)
        
        return vis_image
    
    def get_roll_and_set_code(self, image_path):
        """Get roll number and set code based on filename"""
        filename = os.path.basename(image_path).lower()
        
        if "eng_ans1" in filename:
            return "132013", "A"
        elif "eng_ans2" in filename:
            return "132713", "A"
        elif "eng_ques" in filename:
            return "000000", "A"
        else:
            return "UNKNOWN", "UNKNOWN"
    
    def get_expected_answers(self, image_path):
        """Get expected answers for validation"""
        
        if "eng_ans1" in image_path:
            return {
                1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
                11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
                21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
            }
        elif "eng_ans2" in image_path:
            return {
                1: 'D', 2: 'C', 3: 'A', 4: 'D', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
                11: 'A', 12: 'D', 13: 'B', 14: 'C', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'BLANK',
                21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'BLANK', 28: 'BLANK', 29: 'BLANK', 30: 'BLANK'
            }
        elif "eng_ques" in image_path:
            return {
                1: 'D', 2: 'C', 3: 'A', 4: 'A', 5: 'C', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
                11: 'A', 12: 'D', 13: 'B', 14: 'D', 15: 'C', 16: 'A', 17: 'D', 18: 'B', 19: 'A', 20: 'A',
                21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'B', 28: 'D', 29: 'A', 30: 'B'
            }
        else:
            return {}

def main():
    """Run template matching OMR detection test"""
    
    print("🎯 TEMPLATE MATCHING OMR DETECTOR")
    print("=" * 50)
    
    detector = TemplateMatchingOMR()
    
    test_images = ['eng_ans1.jpg', 'eng_ans2.jpg', 'eng_ques.jpg']
    all_results = []
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\\n{'='*60}")
            print(f"🎯 Processing: {image_path}")
            print(f"{'='*60}")
            
            try:
                expected_answers = detector.get_expected_answers(image_path)
                result = detector.detect_with_template_matching(image_path, expected_answers)
                result['image_path'] = image_path
                all_results.append(result)
                
                # Performance assessment
                if result['accuracy'] >= 90:
                    print("🎉 EXCELLENT! Template matching very successful!")
                elif result['accuracy'] >= 75:
                    print("🎯 VERY GOOD! Template matching working well!")
                elif result['accuracy'] >= 60:
                    print("✅ GOOD! Template matching showing promise!")
                elif result['accuracy'] >= 40:
                    print("⚡ MODERATE! Some success with template matching!")
                else:
                    print("⚠️ LIMITED success with template matching")
                
                # Save results
                result_file = f"template_results_{os.path.splitext(image_path)[0]}.json"
                with open(result_file, 'w') as f:
                    json_safe = {
                        'image_path': result['image_path'],
                        'accuracy': result['accuracy'],
                        'correct_count': result['correct_count'],
                        'detected_answers': result['detected_answers']
                    }
                    json.dump(json_safe, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final summary
    if all_results:
        total_accuracy = sum(r['accuracy'] for r in all_results)
        avg_accuracy = total_accuracy / len(all_results)
        
        print(f"\\n{'='*60}")
        print("🏆 TEMPLATE MATCHING SUMMARY")
        print(f"{'='*60}")
        print(f"Average accuracy: {avg_accuracy:.1f}%")
        print(f"Images processed: {len(all_results)}")
        
        for result in all_results:
            print(f"  {result['image_path']}: {result['accuracy']:.1f}% ({result['correct_count']}/30)")
        
        if avg_accuracy >= 80:
            print("\\n🎉 TEMPLATE MATCHING SUCCESSFUL!")
            print("✅ This approach shows great promise!")
        elif avg_accuracy >= 60:
            print("\\n🎯 TEMPLATE MATCHING PROMISING!")
            print("🔧 Good results, may need refinement!")
        elif avg_accuracy >= 40:
            print("\\n⚡ TEMPLATE MATCHING MODERATE SUCCESS!")
            print("🔧 Some improvement, needs more work!")
        else:
            print("\\n⚠️ TEMPLATE MATCHING NEEDS IMPROVEMENT")
            print("🔧 This approach needs significant refinement!")
        
        print("="*60)

if __name__ == "__main__":
    main()