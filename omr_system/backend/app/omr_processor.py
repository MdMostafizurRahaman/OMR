import cv2
import numpy as np
from .universal_omr_detector_new import UniversalOMRDetector
from .bangladeshi_omr_detector import BangladeshiOMRDetector
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
        self.universal_detector = UniversalOMRDetector()
        self.bangladeshi_detector = BangladeshiOMRDetector()
    
    def process_omr(self, file_path: str) -> Dict:
        """
        Process OMR sheet using dynamic detection optimized for 25 questions
        """
        try:
            # Load and validate image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Enhance image for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
            
            print(f"Processing OMR file: {file_path}")
            print(f"Image shape: {image.shape}")
            
            # Use enhanced image for text extraction
            text_data = self.extract_text(enhanced)
            print(f"Extracted text data: {text_data}")
            
            # Extract roll number and set code with better accuracy
            roll_number = text_data.get('roll_number', '')
            set_code = text_data.get('set_code', 'A')  # Default to A for 25-question format
            
            print(f"Detected roll number: {roll_number}")
            print(f"Detected set code: {set_code}")
            
            # Configure detector for 25 questions - try Bangladeshi detector first
            print("Using Bangladeshi OMR detector optimized for 25 questions...")
            
            # Try Bangladeshi detector first (best for 25-question format)
            answers = self.bangladeshi_detector.detect_omr_answers(file_path)
            
            print(f"Bangladeshi detection result: {len(answers)} answers found")
            
            # If Bangladeshi detector fails, try final detector
            if not answers or len(answers) < 15:  # Need at least 15 answers for 25 questions
                print("Bangladeshi detection insufficient, trying final detector...")
                try:
                    from .final_omr_detector import FinalOMRDetector
                    final_detector = FinalOMRDetector(debug_mode=True)
                    answers = final_detector.process(file_path)
                    print(f"Final detector result: {len(answers)} answers found")
                except ImportError as ie:
                    print(f"Final detector import failed: {ie}")
                    answers = {}
                except Exception as e:
                    print(f"Final detector failed: {e}")
                    answers = {}
            
            # If still insufficient, try universal detector
            if not answers or len(answers) < 10:
                print("Final detection insufficient, trying universal detector...")
                
                # Pre-process image for better detection
                processed = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Save enhanced image temporarily
                import os
                temp_path = os.path.join(os.path.dirname(file_path), "temp_enhanced.jpg")
                cv2.imwrite(temp_path, processed)
                
                # Use enhanced image for detection
                answers = self.universal_detector.detect_omr_answers(temp_path)
                try:
                    os.remove(temp_path)  # Clean up
                except:
                    pass
            
            print(f"Enhanced/Universal detection result: {len(answers)} answers found")
            
            # If still no answers, try visual detector
            if not answers or len(answers) < 10:
                print("Universal detector insufficient, trying visual detector...")
                try:
                    answers = self.visual_detector.process_omr(file_path)
                except:
                    answers = {}
            
            # If still no answers, try smart detector
            if not answers or len(answers) < 5:
                print("Visual detector insufficient, trying smart detector...")
                try:
                    answers = self.smart_detector.process_omr(file_path)
                except:
                    answers = {}
            
            print(f"Final extracted answers: {len(answers)} answers found")
            
            # If still no answers, return basic data structure
            if not answers:
                answers = {}
            
            # Don't auto-assign roll number if it's empty
            final_roll = "" if not roll_number else roll_number
            
            return {
                'roll_number': final_roll,
                'set_code': set_code,
                'answers': answers,
                'total_answers': len(answers),
                'processing_method': 'bangladeshi_detection'
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
            
            # Focus on top-left area where roll number section should be
            roll_area = gray[0:int(height*0.2), 0:int(width*0.3)]
            
            # Apply preprocessing for bubble detection
            blurred = cv2.GaussianBlur(roll_area, (3, 3), 0)
            
            # Try different thresholds for light images
            if np.mean(roll_area) > 200:  # Light image
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of filled bubbles
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_digit_bubbles = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Look for small circular bubbles (digit bubbles are smaller)
                if 10 < area < 300:  # Broader range for different image types
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.4 <= aspect_ratio <= 2.5:  # More lenient aspect ratio
                        filled_digit_bubbles.append({
                            'x': x + w//2,
                            'y': y + h//2,
                            'w': w,
                            'h': h
                        })
            
            if len(filled_digit_bubbles) == 0:
                print("No digit bubbles found for roll number")
                return ""
            
            print(f"Found {len(filled_digit_bubbles)} potential digit bubbles")
            
            # Sort bubbles by position (left to right for columns, top to bottom for digits)
            filled_digit_bubbles.sort(key=lambda b: (b['x'], b['y']))
            
            # Group into columns (each column represents a digit position)
            columns = []
            current_column = []
            
            if len(filled_digit_bubbles) > 0:
                last_x = filled_digit_bubbles[0]['x']
                
                for bubble in filled_digit_bubbles:
                    # If x difference is significant, it's a new column
                    if abs(bubble['x'] - last_x) > 20:  # Reduced threshold
                        if current_column:
                            columns.append(current_column)
                        current_column = [bubble]
                        last_x = bubble['x']
                    else:
                        current_column.append(bubble)
                
                if current_column:
                    columns.append(current_column)
            
            print(f"Organized into {len(columns)} digit columns")
            
            # If no clear columns found, return empty
            if len(columns) == 0:
                return ""
            
            # Extract digits from each column
            roll_digits = []
            
            for col_idx, col in enumerate(columns):
                # Sort by y-coordinate to get digit order (0 at top, 9 at bottom)
                col.sort(key=lambda b: b['y'])
                
                # Find the filled bubble (darkest) in this column
                best_bubble = None
                best_intensity = 255
                
                for i, bubble in enumerate(col):
                    x, y = bubble['x'], bubble['y']
                    roi_size = 3  # Small sampling area
                    x1 = max(0, x - roi_size)
                    y1 = max(0, y - roi_size)
                    x2 = min(roll_area.shape[1], x + roi_size)
                    y2 = min(roll_area.shape[0], y + roi_size)
                    
                    roi = roll_area[y1:y2, x1:x2]
                    if roi.size > 0:
                        mean_intensity = np.mean(roi)
                        if mean_intensity < best_intensity:
                            best_intensity = mean_intensity
                            best_bubble = (i, bubble)
                
                # If we found a filled bubble, add the digit
                if best_bubble and best_intensity < 180:  # Threshold for filled
                    digit_position, bubble_info = best_bubble
                    if digit_position < 10:  # Valid digit (0-9)
                        roll_digits.append(str(digit_position))
                        print(f"Column {col_idx}: digit {digit_position} (intensity: {best_intensity:.1f})")
            
            roll_number = ''.join(roll_digits) if len(roll_digits) >= 3 else ""  # Need at least 3 digits
            print(f"Final extracted roll number: '{roll_number}'")
            
            return roll_number
            
        except Exception as e:
            print(f"Roll number extraction error: {e}")
            return ""
            
        except Exception as e:
            print(f"Roll number extraction error: {e}")
            return ""
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
        """Dynamic OMR detection that adapts to actual question count"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                debug_image = image.copy()
            else:
                gray = image.copy()
                debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            height, width = gray.shape
            print(f"Dynamic OMR detection: {width}x{height}")
            
            # First, detect actual bubble layout
            print("Step 1: Detecting actual bubble layout...")
            bubble_positions = self.detect_actual_bubbles(gray)
            
            if not bubble_positions:
                print("No bubbles detected, falling back to grid detection")
                return self.grid_fallback_detection(gray, debug_image)
            
            print(f"Found {len(bubble_positions)} bubble positions")
            
            # Group bubbles into questions
            questions = self.group_bubbles_into_questions(bubble_positions)
            print(f"Organized into {len(questions)} questions")
            
            # Extract answers from each question
            answers = {}
            for question_num, bubble_group in questions.items():
                if len(bubble_group) >= 4:  # Should have A, B, C, D
                    # Sort by X position for A, B, C, D order
                    bubble_group.sort(key=lambda b: b['x'])
                    
                    bubble_scores = []
                    for opt_idx, bubble in enumerate(bubble_group[:4]):
                        option_letter = chr(65 + opt_idx)  # A, B, C, D
                        
                        # Sample around bubble position
                        x, y = bubble['x'], bubble['y']
                        intensity_score = self.calculate_bubble_intensity(gray, x, y)
                        
                        bubble_scores.append((option_letter, intensity_score, x, y))
                        
                        # Debug visualization
                        color = (0, 255, 0) if intensity_score < 100 else (0, 0, 255)
                        cv2.circle(debug_image, (x, y), 5, color, 2)
                    
                    # Find darkest (filled) bubble
                    if bubble_scores:
                        bubble_scores.sort(key=lambda x: x[1])  # Sort by intensity
                        best = bubble_scores[0]
                        second_best = bubble_scores[1] if len(bubble_scores) > 1 else None
                        
                        filled_option, intensity, bubble_x, bubble_y = best
                        intensity_diff = second_best[1] - best[1] if second_best else 50
                        
                        # Determine if bubble is filled
                        is_filled = (
                            intensity < 90 or  # Lower threshold for dark bubbles
                            (intensity < 130 and intensity_diff > 20) or  # Moderate dark with good separation
                            (intensity < 150 and intensity_diff > 40)     # Light but with excellent separation
                        )
                        
                        if is_filled:
                            answers[str(question_num)] = filled_option
                            print(f"✓ Q{question_num}: {filled_option} (intensity={intensity:.1f}, diff={intensity_diff:.1f})")
                            
                            # Mark answer on debug
                            cv2.circle(debug_image, (bubble_x, bubble_y), 8, (0, 255, 0), 3)
                            cv2.putText(debug_image, f"Q{question_num}:{filled_option}", 
                                      (bubble_x - 15, bubble_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Save debug image
            try:
                debug_path = "d:/OMR/omr_system/backend/debug_dynamic_detection.jpg"
                cv2.imwrite(debug_path, debug_image)
                print(f"Debug image saved to: {debug_path}")
            except:
                pass
            
            print(f"DYNAMIC DETECTION: Found {len(answers)} answers for {len(questions)} questions")
            return answers
            
        except Exception as e:
            print(f"Dynamic detection error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def detect_actual_bubbles(self, gray):
        """Detect actual bubble positions in the OMR"""
        height, width = gray.shape
        
        # Enhanced preprocessing
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use multiple thresholding methods
        _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_or(binary1, adaptive)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - bubbles should be reasonably sized
            if 30 < area < 800:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = float(w) / h
                if 0.4 <= aspect_ratio <= 2.5:
                    
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.3:  # Reasonably circular
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # Only consider bubbles in answer area
                            if center_y > height * 0.25:
                                bubbles.append({
                                    'x': center_x,
                                    'y': center_y,
                                    'w': w,
                                    'h': h,
                                    'area': area,
                                    'circularity': circularity
                                })
        
        return bubbles
    
    def group_bubbles_into_questions(self, bubbles):
        """Group detected bubbles into questions"""
        if not bubbles:
            return {}
        
        # Sort bubbles by Y coordinate to group into rows
        bubbles.sort(key=lambda b: b['y'])
        
        # Group bubbles into rows
        rows = []
        current_row = []
        last_y = bubbles[0]['y']
        
        for bubble in bubbles:
            # If Y difference is significant, start new row
            if abs(bubble['y'] - last_y) > 20:  # Row threshold
                if len(current_row) >= 3:  # Valid row should have multiple bubbles
                    rows.append(sorted(current_row, key=lambda b: b['x']))
                current_row = [bubble]
                last_y = bubble['y']
            else:
                current_row.append(bubble)
        
        # Add last row
        if len(current_row) >= 3:
            rows.append(sorted(current_row, key=lambda b: b['x']))
        
        # Map rows to questions
        questions = {}
        for row_idx, row in enumerate(rows):
            question_num = row_idx + 1
            questions[question_num] = row
        
        return questions
    
    def calculate_bubble_intensity(self, gray, x, y):
        """Calculate intensity score for a bubble at given position"""
        height, width = gray.shape
        
        # Sample in multiple sizes for robustness
        scores = []
        for radius in [4, 6, 8, 10]:
            x1 = max(0, x - radius)
            x2 = min(width, x + radius)
            y1 = max(0, y - radius)
            y2 = min(height, y + radius)
            
            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                avg_intensity = np.mean(roi)
                min_intensity = np.min(roi)
                
                # Count very dark pixels
                very_dark_ratio = np.sum(roi < 80) / roi.size
                
                # Combined score
                score = avg_intensity * 0.6 + min_intensity * 0.4 - (very_dark_ratio * 50)
                scores.append(score)
        
        return np.mean(scores) if scores else 255
    
    def grid_fallback_detection(self, gray, debug_image):
        """Fallback grid detection for difficult cases"""
        print("Using grid fallback detection...")
        height, width = gray.shape
        answers = {}
        
        # Prioritize 25-question layouts for this OMR format
        possible_layouts = [
            {"cols": 1, "questions_per_col": 25},  # 25 questions in 1 column
            {"cols": 2, "questions_per_col": 13},  # ~25 questions in 2 columns
            {"cols": 3, "questions_per_col": 9},   # ~25 questions in 3 columns
            {"cols": 5, "questions_per_col": 5}    # 25 questions in 5 columns
        ]
        
        best_layout = None
        max_answers = 0
        
        for layout in possible_layouts:
            layout_answers = self.try_grid_layout(gray, layout)
            if len(layout_answers) > max_answers:
                max_answers = len(layout_answers)
                best_layout = layout_answers
        
        return best_layout if best_layout else {}
    
    def try_grid_layout(self, gray, layout):
        """Try a specific grid layout"""
        height, width = gray.shape
        answers = {}
        
        cols = layout["cols"]
        questions_per_col = layout["questions_per_col"]
        
        answer_start_y = int(height * 0.25)
        answer_end_y = int(height * 0.95)
        answer_height = answer_end_y - answer_start_y
        
        col_width = width // cols
        
        for col in range(cols):
            for row in range(questions_per_col):
                question_num = (col * questions_per_col) + row + 1
                
                base_x = col * col_width
                base_y = answer_start_y + int((row + 0.5) * (answer_height / questions_per_col))
                
                option_width = col_width // 4
                
                bubble_scores = []
                for option in range(4):
                    option_letter = chr(65 + option)
                    option_x = base_x + int((option + 0.5) * option_width)
                    
                    intensity = self.calculate_bubble_intensity(gray, option_x, base_y)
                    bubble_scores.append((option_letter, intensity))
                
                if bubble_scores:
                    bubble_scores.sort(key=lambda x: x[1])
                    best = bubble_scores[0]
                    
                    if best[1] < 140:  # More lenient threshold for filled bubbles
                        answers[str(question_num)] = best[0]
        
        return answers
    
    def ultra_aggressive_detection(self, gray, debug_image):
        """Ultra-aggressive detection for stubborn OMR sheets"""
        print("Using ultra-aggressive bubble detection...")
        
        height, width = gray.shape
        answers = {}
        
        # Strict grid with multiple sampling points
        answer_start_y = int(height * 0.26)
        answer_end_y = int(height * 0.97)
        answer_height = answer_end_y - answer_start_y
        
        # 4 columns, 25 questions each
        col_boundaries = [
            int(width * 0.05),   # Start of column 1
            int(width * 0.28),   # Start of column 2  
            int(width * 0.51),   # Start of column 3
            int(width * 0.74),   # Start of column 4
            int(width * 0.97)    # End of column 4
        ]
        
        for col_idx in range(4):
            col_start = col_boundaries[col_idx]
            col_end = col_boundaries[col_idx + 1]
            col_width = col_end - col_start
            
            for row in range(25):
                question_num = (col_idx * 25) + row + 1
                if question_num > 100:
                    break
                
                # Calculate row position
                row_y = answer_start_y + int((row + 0.5) * (answer_height / 25))
                
                option_scores = []
                
                # Sample 4 options across the column width
                for opt in range(4):
                    opt_x = col_start + int((opt + 0.5) * (col_width / 4))
                    
                    # Sample multiple points around expected position
                    min_score = 255
                    
                    for dx in range(-8, 9, 4):
                        for dy in range(-8, 9, 4):
                            sample_x = max(0, min(width-1, opt_x + dx))
                            sample_y = max(0, min(height-1, row_y + dy))
                            
                            # Sample in small region
                            roi_size = 3
                            x1 = max(0, sample_x - roi_size)
                            x2 = min(width, sample_x + roi_size)
                            y1 = max(0, sample_y - roi_size)
                            y2 = min(height, sample_y + roi_size)
                            
                            roi = gray[y1:y2, x1:x2]
                            if roi.size > 0:
                                avg_intensity = np.mean(roi)
                                min_score = min(min_score, avg_intensity)
                    
                    option_letter = chr(65 + opt)
                    option_scores.append((option_letter, min_score, opt_x, row_y))
                
                # Find darkest option
                if option_scores:
                    option_scores.sort(key=lambda x: x[1])
                    best_option, best_score, best_x, best_y = option_scores[0]
                    
                    # Very aggressive threshold
                    if best_score < 140:
                        answers[str(question_num)] = best_option
                        print(f"✓ ULTRA Q{question_num}: {best_option} (score={best_score:.1f})")
                        
                        # Mark on debug image
                        cv2.circle(debug_image, (best_x, best_y), 10, (255, 0, 255), 2)
        
        return answers
    
    def fallback_grid_detection(self, gray, debug_image):
        """Fallback method using aggressive grid detection"""
        print("Using fallback grid detection method...")
        height, width = gray.shape
        answers = {}
        
        # Divide into strict grid
        answer_start_y = int(height * 0.32)
        answer_end_y = int(height * 0.94)
        answer_height = answer_end_y - answer_start_y
        
        # 4 columns, 25 rows each
        col_width = width // 4
        row_height = answer_height // 25
        
        for col in range(4):
            for row in range(25):
                question_num = (col * 25) + row + 1
                if question_num > 100:
                    break
                
                base_x = col * col_width
                base_y = answer_start_y + (row * row_height)
                option_width = col_width // 4
                
                bubble_scores = []
                
                for option in range(4):
                    option_letter = chr(65 + option)
                    center_x = base_x + (option * option_width) + (option_width // 2)
                    center_y = base_y + (row_height // 2)
                    
                    # Multiple ROI sizes to catch different bubble sizes
                    for roi_size in [8, 12, 16]:
                        x1, x2 = max(0, center_x - roi_size), min(width, center_x + roi_size)
                        y1, y2 = max(0, center_y - roi_size), min(height, center_y + roi_size)
                        
                        roi = gray[y1:y2, x1:x2]
                        if roi.size > 0:
                            avg_intensity = np.mean(roi)
                            dark_ratio = np.sum(roi < 100) / roi.size
                            score = avg_intensity - (dark_ratio * 80)
                            bubble_scores.append((option_letter, score, roi_size))
                            break
                
                if bubble_scores:
                    bubble_scores.sort(key=lambda x: x[1])
                    best_option, best_score, best_size = bubble_scores[0]
                    
                    # Very aggressive threshold
                    if best_score < 130:
                        answers[str(question_num)] = best_option
                        print(f"✓ FALLBACK Q{question_num}: {best_option} (score={best_score:.1f})")
        
        return answers
    
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