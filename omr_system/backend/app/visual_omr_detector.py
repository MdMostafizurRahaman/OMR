import cv2
import numpy as np
from typing import Dict, List, Tuple

class VisualOMRDetector:
    """
    Visual OMR detector that reads filled dots directly from the image
    Specifically designed for Bengali OMR with filled black dots
    """
    
    def __init__(self):
        self.debug_mode = True
        
    def process_omr(self, image_path: str) -> Dict[str, str]:
        """
        Main processing function that reads the OMR visually
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load image")
                return {}
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            print(f"Processing image: {width}x{height}")
            
            # Extract answers using visual detection
            answers = self.extract_answers_visual(gray)
            
            print(f"Visual detector extracted {len(answers)} answers")
            return answers
            
        except Exception as e:
            print(f"Visual detector error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_answers_visual(self, gray: np.ndarray) -> Dict[str, str]:
        """
        Extract answers by looking for filled black dots in specific locations
        """
        height, width = gray.shape
        
        # Define the answer area (skip header)
        answer_start_y = int(height * 0.30)  # Start below instructions
        answer_end_y = int(height * 0.95)    # End before bottom margin
        answer_height = answer_end_y - answer_start_y
        
        # Bengali OMR layout: 4 columns, 25 questions each
        num_columns = 4
        questions_per_column = 25
        
        # Calculate grid dimensions
        column_width = width // num_columns
        row_height = answer_height // questions_per_column
        
        print(f"Grid layout: {num_columns} columns x {questions_per_column} rows")
        print(f"Cell size: {column_width}x{row_height}")
        
        answers = {}
        
        # Process each question
        for col in range(num_columns):
            for row in range(questions_per_column):
                question_num = (col * questions_per_column) + row + 1
                
                if question_num > 100:
                    break
                
                # Calculate the base position for this question row
                base_x = col * column_width
                base_y = answer_start_y + (row * row_height)
                
                # Find the darkest option (filled bubble) for this question
                answer = self.find_filled_option(gray, base_x, base_y, column_width, row_height, question_num)
                
                if answer:
                    answers[str(question_num)] = answer
        
        return answers
    
    def find_filled_option(self, gray: np.ndarray, base_x: int, base_y: int, 
                          col_width: int, row_height: int, question_num: int) -> str:
        """
        Find which option (A, B, C, D) is filled for a specific question
        """
        options = ['A', 'B', 'C', 'D']
        option_width = col_width // 4  # 4 options per question
        
        option_scores = []
        
        for i, option in enumerate(options):
            # Calculate center position for this option
            option_x = base_x + (i * option_width) + (option_width // 2)
            option_y = base_y + (row_height // 2)
            
            # Define sampling area around the expected bubble position
            sample_size = min(12, option_width // 3, row_height // 3)
            
            x_start = max(0, option_x - sample_size)
            x_end = min(gray.shape[1], option_x + sample_size)
            y_start = max(0, option_y - sample_size)
            y_end = min(gray.shape[0], option_y + sample_size)
            
            # Extract the region
            roi = gray[y_start:y_end, x_start:x_end]
            
            if roi.size > 0:
                # Calculate darkness metrics
                avg_intensity = np.mean(roi)
                min_intensity = np.min(roi)
                
                # Count very dark pixels (filled bubble indicator)
                very_dark_count = np.sum(roi < 80)  # Very dark pixels
                dark_count = np.sum(roi < 120)      # Dark pixels
                total_pixels = roi.size
                
                # Calculate dark ratios
                very_dark_ratio = very_dark_count / total_pixels
                dark_ratio = dark_count / total_pixels
                
                # Darkness score (higher = more filled)
                # Prioritize very dark pixels and low average intensity
                darkness_score = (very_dark_ratio * 100) + (dark_ratio * 50) + ((255 - avg_intensity) / 255 * 50)
                
                option_scores.append({
                    'option': option,
                    'score': darkness_score,
                    'avg_intensity': avg_intensity,
                    'very_dark_ratio': very_dark_ratio,
                    'dark_ratio': dark_ratio,
                    'min_intensity': min_intensity
                })
                
                if self.debug_mode and question_num <= 5:  # Debug first few questions
                    print(f"Q{question_num}{option}: pos({option_x},{option_y}) "
                          f"avg={avg_intensity:.1f} very_dark={very_dark_ratio:.2f} "
                          f"dark={dark_ratio:.2f} score={darkness_score:.1f}")
        
        # Find the option with highest darkness score
        if not option_scores:
            return None
        
        # Sort by darkness score (descending)
        option_scores.sort(key=lambda x: x['score'], reverse=True)
        
        best = option_scores[0]
        second_best = option_scores[1] if len(option_scores) > 1 else None
        
        # Quality checks
        score_difference = 0
        if second_best:
            score_difference = best['score'] - second_best['score']
        
        # Accept answer if:
        # 1. High darkness score
        # 2. Significant very dark pixel ratio
        # 3. Clear difference from second option
        if (best['score'] > 30 and                    # Minimum darkness threshold
            best['very_dark_ratio'] > 0.2 and         # At least 20% very dark pixels
            best['avg_intensity'] < 150 and           # Average not too bright
            score_difference > 15):                   # Clear winner
            
            if self.debug_mode and question_num <= 10:
                print(f"✓ Q{question_num}: {best['option']} "
                      f"(score: {best['score']:.1f}, diff: {score_difference:.1f})")
            
            return best['option']
        else:
            if self.debug_mode and question_num <= 10:
                print(f"? Q{question_num}: No clear answer "
                      f"(best: {best['score']:.1f}, diff: {score_difference:.1f})")
            
            return None