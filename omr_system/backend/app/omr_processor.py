import cv2
import numpy as np
import re
import json
from typing import Dict, List, Optional
from .smart_omr_detector import SmartOMRDetector
from .visual_omr_detector import VisualOMRDetector

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    # Try to configure tesseract path for Windows
    try:
        import subprocess
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
    except:
        # Try common Windows paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\Public\Tesseract-OCR\tesseract.exe',
            r'C:\tesseract\tesseract.exe',
            r'D:\tesseract\tesseract.exe'
        ]
        for path in possible_paths:
            try:
                import os
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
            except:
                continue
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR features will be limited.")

class OMRProcessor:
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.smart_detector = SmartOMRDetector()
        self.visual_detector = VisualOMRDetector()
    
    def process_omr(self, file_path: str) -> Dict:
        """
        Process OMR sheet and extract data
        Returns dictionary with extracted information
        """
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")
            
            print(f"Processing OMR file: {file_path}")
            print(f"Image shape: {image.shape}")
            
            # Extract text using OCR
            text_data = self.extract_text(image)
            print(f"Extracted text data: {text_data}")
            
            # Extract roll number and set code from the returned dict
            roll_number = text_data.get('roll_number', '')
            set_code = text_data.get('set_code', 'A')
            
            print(f"Detected roll number: {roll_number}")
            print(f"Detected set code: {set_code}")
            
            # Try the new visual detector first (best for your OMR format)
            print("Trying visual OMR detection...")
            answers = self.visual_detector.process_omr(file_path)
            
            # If visual detector doesn't get enough answers, try smart detector
            if not answers or len(answers) < 80:
                print("Visual detector insufficient, trying smart detector...")
                answers = self.smart_detector.process_omr(file_path)
            
            # If still not enough answers, try fallback methods
            if not answers or len(answers) < 50:
                print("Smart detector failed, trying fallback methods...")
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
                
                # Method 1: Template-based detection
                print("Trying template-based detection...")
                answers = self.extract_answers_from_bubbles(processed_image)
                
                # Method 2: Try with original image if first method fails
                if not answers:
                    print("Trying with original image...")
                    answers = self.extract_answers_from_bubbles(image)
                
                # Method 3: Try with different preprocessing
                if not answers:
                    print("Trying alternative preprocessing...")
                    alt_processed = self.alternative_preprocessing(image)
                    answers = self.extract_answers_from_bubbles(alt_processed)
            
            print(f"Final extracted answers: {len(answers)} answers found")
            
            # If still no answers, return basic data structure
            if not answers:
                answers = {}
            
            return {
                'roll_number': roll_number,
                'set_code': set_code,
                'answers': answers,
                'total_answers': len(answers),
                'processing_method': 'visual_detector' if len(answers) >= 80 else ('smart_detector' if len(answers) >= 50 else 'fallback')
            }
            
        except Exception as e:
            print(f"Error processing OMR: {str(e)}")
            # Fallback: return structure for manual entry
            return {
                "roll_number": None,
                "set_code": None,
                "answers": {},
                "total_answers": 0,
                "extracted_text": f"Error: {str(e)}",
                "processing_method": "manual_entry_required"
            }
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR and bubble detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image) -> Dict[str, str]:
        """Extract roll number and set code from Bengali OMR format"""
        try:
            # Create grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            
            # Extract roll number from the top-left section
            roll_number = self.extract_roll_number_bengali(gray)
            
            # Extract set code from top section
            set_code = self.extract_set_code_bengali(gray)
            
            return {
                'roll_number': roll_number,
                'set_code': set_code
            }
            
        except Exception as e:
            print(f"Text extraction error: {e}")
            return {'roll_number': '', 'set_code': ''}
    
    def extract_roll_number_bengali(self, gray):
        """Extract roll number from filled digit bubbles"""
        try:
            height, width = gray.shape
            
            # Focus on top-left area where roll number section is
            roll_area = gray[0:int(height*0.3), 0:int(width*0.4)]
            
            # Apply preprocessing for bubble detection
            blurred = cv2.GaussianBlur(roll_area, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of filled bubbles
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_digit_bubbles = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Look for small circular bubbles (digit bubbles are smaller)
                if 20 < area < 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.6 <= aspect_ratio <= 1.4:
                        filled_digit_bubbles.append({
                            'x': x + w//2,
                            'y': y + h//2,
                            'w': w,
                            'h': h
                        })
            
            if len(filled_digit_bubbles) == 0:
                return ""
            
            # Sort bubbles by position (left to right for columns, top to bottom for digits)
            filled_digit_bubbles.sort(key=lambda b: (b['x'], b['y']))
            
            # Group into columns (each column represents a digit position)
            columns = []
            current_column = []
            
            if len(filled_digit_bubbles) > 0:
                last_x = filled_digit_bubbles[0]['x']
                
                for bubble in filled_digit_bubbles:
                    # If x difference is significant, it's a new column
                    if abs(bubble['x'] - last_x) > 30:
                        if current_column:
                            columns.append(current_column)
                        current_column = [bubble]
                        last_x = bubble['x']
                    else:
                        current_column.append(bubble)
                
                if current_column:
                    columns.append(current_column)
            
            # Extract digits from each column
            roll_digits = []
            
            for col in columns:
                # Sort by y-coordinate to get digit order (0 at top, 9 at bottom)
                col.sort(key=lambda b: b['y'])
                
                # Find the filled bubble (darkest) in this column
                if len(col) > 0:
                    # For now, take the first one (you may need to improve this)
                    # In a proper implementation, you'd check which bubble is actually filled
                    digit_position = min(len(col) - 1, 9)  # Ensure it's 0-9
                    roll_digits.append(str(digit_position))
            
            roll_number = ''.join(roll_digits) if roll_digits else ""
            print(f"Extracted roll number: {roll_number}")
            
            return roll_number
            
        except Exception as e:
            print(f"Roll number extraction error: {e}")
            return ""
    
    def extract_set_code_bengali(self, gray):
        """Extract set code from the set code section"""
        try:
            height, width = gray.shape
            
            # Focus on the set code area (top center area)
            set_area = gray[0:int(height*0.25), int(width*0.3):int(width*0.7)]
            
            # Apply preprocessing
            blurred = cv2.GaussianBlur(set_area, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_bubbles = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 50 < area < 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.6 <= aspect_ratio <= 1.4:
                        filled_bubbles.append({
                            'x': x + w//2,
                            'y': y + h//2,
                            'area': area
                        })
            
            if len(filled_bubbles) == 0:
                return "A"  # Default
            
            # Sort by x-coordinate (A, B, C, D from left to right)
            filled_bubbles.sort(key=lambda b: b['x'])
            
            # The leftmost filled bubble determines the set code
            # This is a simplified approach - you may need to refine
            if len(filled_bubbles) > 0:
                # For now, return A (you can improve this by analyzing position)
                return "A"
            
            return "A"
            
        except Exception as e:
            print(f"Set code extraction error: {e}")
            return "A"
    
    def extract_text_fallback(self, image):
        """Fallback text extraction without tesseract"""
        return "Tesseract not available - using pattern matching only"
    
    def extract_roll_number(self, text: str) -> Optional[str]:
        """Extract roll number from text - ENHANCED PATTERNS"""
        print(f"Searching for roll number in text: {text[:200]}...")
        
        # Enhanced patterns for roll numbers
        patterns = [
            # Bangladeshi patterns
            r'Roll\s*No[:\.\-\s]*(\d+)',
            r'Roll\s*Number[:\.\-\s]*(\d+)', 
            r'Roll[:\.\-\s]*(\d+)',
            r'রোল[:\.\-\s]*(\d+)',  # Bengali
            
            # General patterns
            r'ID[:\.\-\s]*(\d+)',
            r'Student\s*ID[:\.\-\s]*(\d+)',
            r'Reg[:\.\-\s]*(\d+)',
            r'Registration[:\.\-\s]*(\d+)',
            r'Admission[:\.\-\s]*(\d+)',
            
            # Number patterns
            r'(\d{6,12})',  # 6-12 digit numbers
            r'(\d{4,8})',   # 4-8 digit numbers
            
            # Common OMR patterns
            r'No[:\.\-\s]*(\d+)',
            r'Number[:\.\-\s]*(\d+)',
        ]
        
        all_matches = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Filter reasonable roll numbers (not too short/long)
                if 4 <= len(match) <= 12:
                    all_matches.append(match)
                    print(f"Found potential roll number: {match} (pattern: {pattern})")
        
        if all_matches:
            # Return the most likely candidate (longest reasonable number)
            best_match = max(all_matches, key=len)
            print(f"Selected roll number: {best_match}")
            return best_match
        
        print("No roll number found")
        return None
    
    def extract_set_code(self, text: str) -> Optional[str]:
        """Extract set code from text - ENHANCED PATTERNS"""
        print(f"Searching for set code in text: {text[:200]}...")
        
        patterns = [
            # Common English patterns
            r'Set[:\.\-\s]*([A-Z])',
            r'Code[:\.\-\s]*([A-Z])', 
            r'Version[:\.\-\s]*([A-Z])',
            r'Paper[:\.\-\s]*([A-Z])',
            r'Series[:\.\-\s]*([A-Z])',
            r'\b([A-Z])\s*Set\b',
            r'\bSet\s*([A-Z])\b',
            
            # Bengali patterns
            r'সেট[:\.\-\s]*([A-Z])',
            
            # Pattern variations
            r'Question\s*Set[:\.\-\s]*([A-Z])',
            r'Form[:\.\-\s]*([A-Z])',
            r'\b([A-Z])\s*Form\b',
            
            # Just single letters near common words
            r'(?:Set|Code|Version|Paper|সেট)\s*[:\.\-]?\s*([A-Z])',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result = matches[0].upper()
                print(f"Found set code: {result} (pattern: {pattern})")
                return result
        
        print("No set code found")
        return None
    
    def extract_answers_from_bubbles(self, image) -> Dict[str, str]:
        """TEMPLATE-BASED OMR EXTRACTION - Designed for exact Bengali OMR layout"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            print(f"Processing image: {width}x{height}")
            
            # Based on your OMR images, create a precise template for bubble locations
            # Your OMR has specific coordinates where bubbles should be
            
            # Skip header area - answer section starts around 30% down
            answer_start_y = int(height * 0.30)
            answer_end_y = int(height * 0.95)  # Leave some bottom margin
            
            answer_height = answer_end_y - answer_start_y
            
            # Your OMR has 4 main columns for question groups
            # Column 1: Questions 1-25, Column 2: 26-50, Column 3: 51-75, Column 4: 76-100
            
            answers = {}
            
            # Define the approximate layout based on your OMR format
            questions_per_column = 25
            options_per_question = 4  # A, B, C, D
            
            # Calculate spacing
            column_width = width // 4
            row_height = answer_height // questions_per_column
            
            print(f"Grid: {column_width}x{row_height} per cell")
            
            for col in range(4):  # 4 columns
                for row in range(questions_per_column):  # 25 questions per column
                    question_num = (col * questions_per_column) + row + 1
                    
                    if question_num > 100:
                        break
                    
                    # Calculate base position for this question
                    base_x = col * column_width
                    base_y = answer_start_y + (row * row_height)
                    
                    # Within each question area, find the 4 option bubbles (A, B, C, D)
                    option_width = column_width // 4
                    
                    best_option = None
                    min_intensity = 255  # Start with white (highest intensity)
                    
                    for option in range(4):  # A, B, C, D
                        option_letter = chr(65 + option)
                        
                        # Calculate bubble center position
                        bubble_x = base_x + (option * option_width) + (option_width // 2)
                        bubble_y = base_y + (row_height // 2)
                        
                        # Define a small region around the expected bubble location
                        bubble_size = min(20, option_width // 3, row_height // 3)
                        
                        x_start = max(0, bubble_x - bubble_size)
                        x_end = min(width, bubble_x + bubble_size)
                        y_start = max(0, bubble_y - bubble_size)
                        y_end = min(height, bubble_y + bubble_size)
                        
                        # Extract the region where the bubble should be
                        roi = gray[y_start:y_end, x_start:x_end]
                        
                        if roi.size > 0:
                            # Calculate average intensity in this region
                            avg_intensity = np.mean(roi)
                            
                            # Also check minimum intensity to catch dark filled bubbles
                            min_pixel = np.min(roi)
                            
                            # Combined score - lower is more likely to be filled
                            fill_score = (avg_intensity * 0.7) + (min_pixel * 0.3)
                            
                            print(f"Q{question_num}{option_letter}: pos({bubble_x},{bubble_y}) score={fill_score:.1f}")
                            
                            # Track the darkest (most likely filled) bubble
                            if fill_score < min_intensity:
                                min_intensity = fill_score
                                best_option = option_letter
                    
                    # Only record an answer if we found a significantly darker bubble
                    if best_option and min_intensity < 140:  # Threshold for filled bubble
                        answers[str(question_num)] = best_option
                        print(f"✓ Q{question_num}: {best_option} (score: {min_intensity:.1f})")
            
            print(f"Extracted {len(answers)} answers using template matching")
            return answers
            
        except Exception as e:
            print(f"Template matching error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_answers_from_text(self, text: str) -> Dict[str, str]:
        """Fallback: Extract answers from OCR text"""
        answers = {}
        
        # Look for patterns like "1. A", "2. B", etc.
        pattern = r'(\d+)[\.\s]*([A-D])'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for question, answer in matches:
            answers[question] = answer.upper()
        
        return answers
    
    def manual_entry_template(self, total_questions: int = 100) -> Dict[str, str]:
        """Generate template for manual answer entry"""
        return {str(i): "" for i in range(1, total_questions + 1)}
    
    def alternative_preprocessing(self, image):
        """Alternative preprocessing method for difficult images"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply median filter to reduce noise
        filtered = cv2.medianBlur(enhanced, 3)
        
        # Apply binary threshold
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_answers_template_matching(self, image) -> Dict[str, str]:
        """Extract answers using template matching approach"""
        try:
            # This is a simplified template matching approach
            # In a real implementation, you would have templates for filled/unfilled bubbles
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours and filter by size and shape
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            answers = {}
            question_count = 0
            
            # Simple approach: assume every 4 contours represent A, B, C, D for a question
            contour_groups = []
            for i in range(0, len(contours), 4):
                group = contours[i:i+4]
                if len(group) == 4:
                    contour_groups.append(group)
            
            for group_idx, group in enumerate(contour_groups):
                question_num = group_idx + 1
                
                # Check each contour in the group
                for option_idx, contour in enumerate(group):
                    area = cv2.contourArea(contour)
                    
                    # If area is above threshold, consider it filled
                    if area > 100:  # Adjust threshold as needed
                        option = chr(65 + option_idx)  # A, B, C, D
                        answers[str(question_num)] = option
                        break
            
            print(f"Template matching found {len(answers)} answers")
            return answers
            
        except Exception as e:
            print(f"Template matching error: {e}")
            return {}