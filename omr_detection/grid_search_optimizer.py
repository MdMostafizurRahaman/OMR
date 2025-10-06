"""
SYSTEMATIC GRID SEARCH OPTIMIZER
Find exact bubble positions through grid search optimization
"""

import cv2
import numpy as np
import json
import os
from itertools import product

class GridSearchOptimizer:
    def __init__(self):
        self.debug = True
        
    def systematic_position_search(self, image_path):
        """Find optimal bubble positions through systematic grid search"""
        
        print(f"\\n🔍 GRID SEARCH OPTIMIZATION: {image_path}")
        print("="*70)
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Enhanced preprocessing
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Get expected answers for optimization
        expected_answers = self.get_expected_answers(image_path)
        
        print(f"Image size: {width}x{height}")
        print(f"Expected answers loaded: {len(expected_answers)} questions")
        
        # Grid search parameters
        search_ranges = {
            'col1_left': range(70, 110, 5),        # Search column 1 left position
            'col2_left': range(230, 270, 5),       # Search column 2 left position  
            'col3_left': range(390, 430, 5),       # Search column 3 left position
            'first_row_top': range(280, 320, 5),   # Search first row top position
            'row_spacing': range(30, 40, 2),       # Search row spacing
            'bubble_spacing': range(15, 25, 2),    # Search bubble spacing
            'bubble_offset_x': range(20, 35, 2),   # Search bubble offset
        }
        
        print("Starting systematic grid search...")
        print(f"Search space size: {np.prod([len(r) for r in search_ranges.values()]):,} combinations")
        
        best_config = None
        best_accuracy = 0
        best_results = None
        
        # Reduce search space for feasibility
        sample_configs = self.generate_sample_configurations(search_ranges, max_samples=200)
        
        print(f"Testing {len(sample_configs)} sample configurations...")
        
        for i, config in enumerate(sample_configs):
            if i % 20 == 0:
                print(f"Progress: {i+1}/{len(sample_configs)} ({(i+1)/len(sample_configs)*100:.1f}%)")
            
            try:
                # Generate bubble positions with this configuration
                positions = self.generate_positions_from_config(config)
                
                # Test accuracy with these positions
                accuracy, detected_answers = self.test_configuration(enhanced, positions, expected_answers)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config.copy()
                    best_results = detected_answers.copy()
                    print(f"New best configuration found! Accuracy: {accuracy:.1f}%")
                    print(f"Config: {config}")
                
            except Exception as e:
                continue  # Skip invalid configurations
        
        print(f"\\n🎯 OPTIMIZATION COMPLETE!")
        print(f"Best accuracy achieved: {best_accuracy:.1f}%")
        print(f"Best configuration: {best_config}")
        
        if best_config:
            # Generate final visualization with best configuration
            final_positions = self.generate_positions_from_config(best_config)
            self.create_visualization(image, enhanced, final_positions, best_results, expected_answers, image_path)
        
        return {
            'accuracy': best_accuracy,
            'configuration': best_config,
            'detected_answers': best_results,
            'positions': final_positions if best_config else None
        }
    
    def generate_sample_configurations(self, search_ranges, max_samples=200):
        """Generate sample configurations for grid search"""
        
        # Generate random samples from search space
        configs = []
        
        for _ in range(max_samples):
            config = {}
            for param, param_range in search_ranges.items():
                config[param] = np.random.choice(param_range)
            configs.append(config)
        
        # Also include some systematic samples
        systematic_configs = [
            # Default configuration
            {'col1_left': 85, 'col2_left': 245, 'col3_left': 405, 'first_row_top': 295, 
             'row_spacing': 34, 'bubble_spacing': 20, 'bubble_offset_x': 25},
            # Shifted configurations
            {'col1_left': 80, 'col2_left': 240, 'col3_left': 400, 'first_row_top': 290,
             'row_spacing': 32, 'bubble_spacing': 18, 'bubble_offset_x': 22},
            {'col1_left': 90, 'col2_left': 250, 'col3_left': 410, 'first_row_top': 300,
             'row_spacing': 36, 'bubble_spacing': 22, 'bubble_offset_x': 28},
        ]
        
        configs.extend(systematic_configs)
        
        return configs
    
    def generate_positions_from_config(self, config):
        """Generate bubble positions from configuration"""
        
        positions = {}
        
        columns = [
            {'start_q': 1, 'left': config['col1_left']},
            {'start_q': 11, 'left': config['col2_left']},
            {'start_q': 21, 'left': config['col3_left']},
        ]
        
        for col_info in columns:
            col_left = col_info['left']
            start_q = col_info['start_q']
            
            for row in range(10):
                question_num = start_q + row
                row_y = config['first_row_top'] + row * config['row_spacing']
                
                question_bubbles = {}
                for i, option in enumerate(['A', 'B', 'C', 'D']):
                    bubble_x = col_left + config['bubble_offset_x'] + i * config['bubble_spacing']
                    bubble_y = row_y
                    question_bubbles[option] = (bubble_x, bubble_y)
                
                positions[question_num] = question_bubbles
        
        return positions
    
    def test_configuration(self, enhanced, positions, expected_answers):
        """Test a configuration and return accuracy"""
        
        detected_answers = {}
        correct_count = 0
        
        for question_num in range(1, 31):
            if question_num not in positions:
                continue
                
            question_bubbles = positions[question_num]
            option_scores = {}
            
            for option in ['A', 'B', 'C', 'D']:
                bubble_x, bubble_y = question_bubbles[option]
                
                # Quick scoring method for grid search
                scores = []
                
                # Sample around bubble position
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        px, py = bubble_x + dx, bubble_y + dy
                        if 0 <= px < enhanced.shape[1] and 0 <= py < enhanced.shape[0]:
                            val = enhanced[py, px]
                            scores.append(255 - val)
                
                final_score = np.mean(scores) if scores else 0
                option_scores[option] = final_score
            
            # Determine answer
            max_score = max(option_scores.values())
            if max_score > 60:  # Threshold for detection
                detected_option = max(option_scores, key=option_scores.get)
                detected_answers[question_num] = detected_option
            else:
                detected_answers[question_num] = "BLANK"
            
            # Check accuracy
            expected = expected_answers.get(question_num, "BLANK")
            detected = detected_answers[question_num]
            if detected == expected:
                correct_count += 1
        
        accuracy = (correct_count / 30) * 100
        return accuracy, detected_answers
    
    def create_visualization(self, image, enhanced, positions, detected_answers, expected_answers, image_path):
        """Create visualization with optimized positions"""
        
        vis_image = image.copy()
        
        for question_num in range(1, 31):
            if question_num not in positions:
                continue
                
            question_bubbles = positions[question_num]
            
            for option in ['A', 'B', 'C', 'D']:
                bubble_x, bubble_y = question_bubbles[option]
                
                # Check if this option was detected
                detected = detected_answers.get(question_num, "BLANK")
                expected = expected_answers.get(question_num, "BLANK")
                
                is_detected = (detected == option)
                is_correct = (detected == expected)
                
                # Color coding
                if is_detected and is_correct:
                    color = (0, 255, 0)  # Green for correct detection
                    thickness = 3
                elif is_detected and not is_correct:
                    color = (0, 165, 255)  # Orange for incorrect detection
                    thickness = 2
                elif not is_detected and expected == option:
                    color = (0, 0, 255)  # Red for missed detection
                    thickness = 2
                else:
                    color = (128, 128, 128)  # Gray for normal
                    thickness = 1
                
                cv2.circle(vis_image, (bubble_x, bubble_y), 6, color, thickness)
                cv2.putText(vis_image, option, (bubble_x-3, bubble_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Save visualization
        output_filename = f"optimized_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, vis_image)
        print(f"Optimization visualization saved: {output_filename}")
    
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
    """Run grid search optimization"""
    
    print("🔍 SYSTEMATIC GRID SEARCH OPTIMIZER")
    print("=" * 60)
    
    optimizer = GridSearchOptimizer()
    
    test_images = ['eng_ans1.jpg', 'eng_ans2.jpg', 'eng_ques.jpg']
    all_results = []
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\\n{'='*70}")
            print(f"🔍 Optimizing: {image_path}")
            print(f"{'='*70}")
            
            try:
                result = optimizer.systematic_position_search(image_path)
                result['image_path'] = image_path
                all_results.append(result)
                
                # Performance assessment
                if result['accuracy'] >= 90:
                    print("🎉 EXCELLENT! Optimization achieved high accuracy!")
                elif result['accuracy'] >= 75:
                    print("🎯 VERY GOOD! Optimization working well!")
                elif result['accuracy'] >= 60:
                    print("✅ GOOD! Significant improvement achieved!")
                elif result['accuracy'] >= 40:
                    print("⚡ PROGRESS! Optimization showing results!")
                else:
                    print("⚠️ Limited improvement from optimization")
                
                # Save optimization results
                result_file = f"grid_search_results_{os.path.splitext(image_path)[0]}.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'image_path': image_path,
                        'accuracy': result['accuracy'],
                        'configuration': result['configuration'],
                        'detected_answers': result['detected_answers']
                    }, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error optimizing {image_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final optimization summary
    if all_results:
        total_accuracy = sum(r['accuracy'] for r in all_results)
        avg_accuracy = total_accuracy / len(all_results)
        
        print(f"\\n{'='*70}")
        print("🏆 GRID SEARCH OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Average accuracy: {avg_accuracy:.1f}%")
        print(f"Images optimized: {len(all_results)}")
        
        for result in all_results:
            print(f"\\n{result['image_path']}: {result['accuracy']:.1f}%")
            if result['configuration']:
                print(f"  Optimal config: {result['configuration']}")
        
        if avg_accuracy >= 85:
            print("\\n🎉 OPTIMIZATION SUCCESS!")
            print("✅ Grid search achieved excellent results!")
        elif avg_accuracy >= 70:
            print("\\n🎯 OPTIMIZATION EFFECTIVE!")  
            print("✅ Significant improvement through systematic search!")
        elif avg_accuracy >= 55:
            print("\\n⚡ OPTIMIZATION HELPFUL!")
            print("🔧 Good progress, may need fine-tuning!")
        elif avg_accuracy >= 35:
            print("\\n⚠️ MODERATE OPTIMIZATION")
            print("🔧 Some improvement, more work needed!")
        else:
            print("\\n❌ LIMITED OPTIMIZATION SUCCESS")
            print("🔧 Fundamental issues may need addressing!")
        
        print("="*70)

if __name__ == "__main__":
    main()