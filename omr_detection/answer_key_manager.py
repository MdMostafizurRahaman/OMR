"""
Answer Key Management System for OMR Evaluation
Manages correct answers and performs evaluation
"""

import json
import os
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class AnswerKeyManager:
    def __init__(self, data_folder="answer_keys"):
        self.data_folder = data_folder
        self.ensure_data_folder()
        
    def ensure_data_folder(self):
        """Create data folder if it doesn't exist"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"Created answer keys folder: {self.data_folder}")
    
    def create_answer_key(self, exam_name: str, answers: Dict[int, str], 
                         metadata: Optional[Dict] = None) -> str:
        """
        Create a new answer key
        
        Args:
            exam_name: Name/identifier for the exam
            answers: Dictionary mapping question numbers to correct answers
            metadata: Additional information about the exam
        
        Returns:
            Path to the saved answer key file
        """
        if metadata is None:
            metadata = {}
        
        # Add creation timestamp
        metadata['created_at'] = datetime.now().isoformat()
        metadata['total_questions'] = len(answers)
        
        answer_key_data = {
            'exam_name': exam_name,
            'answers': answers,
            'metadata': metadata
        }
        
        # Save to JSON file
        filename = f"{exam_name.replace(' ', '_').lower()}_answer_key.json"
        filepath = os.path.join(self.data_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(answer_key_data, f, indent=2, ensure_ascii=False)
        
        print(f"Answer key saved: {filepath}")
        return filepath
    
    def load_answer_key(self, exam_name_or_path: str) -> Dict:
        """
        Load an answer key by exam name or file path
        
        Args:
            exam_name_or_path: Either exam name or full path to answer key file
        
        Returns:
            Dictionary containing answer key data
        """
        # Check if it's a full path
        if os.path.exists(exam_name_or_path):
            filepath = exam_name_or_path
        else:
            # Try to find by exam name
            filename = f"{exam_name_or_path.replace(' ', '_').lower()}_answer_key.json"
            filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Answer key not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_answer_keys(self) -> List[Dict]:
        """List all available answer keys"""
        answer_keys = []
        
        for filename in os.listdir(self.data_folder):
            if filename.endswith('_answer_key.json'):
                filepath = os.path.join(self.data_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        answer_keys.append({
                            'filename': filename,
                            'exam_name': data.get('exam_name', 'Unknown'),
                            'total_questions': data.get('metadata', {}).get('total_questions', 0),
                            'created_at': data.get('metadata', {}).get('created_at', 'Unknown'),
                            'filepath': filepath
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return sorted(answer_keys, key=lambda x: x.get('created_at', ''))
    
    def create_from_template(self, exam_name: str, total_questions: int = 30) -> str:
        """
        Create an answer key template with all questions set to 'A'
        User can then modify it manually
        """
        answers = {i: 'A' for i in range(1, total_questions + 1)}
        
        metadata = {
            'description': 'Template answer key - Please update with correct answers',
            'is_template': True
        }
        
        return self.create_answer_key(exam_name, answers, metadata)
    
    def create_from_csv(self, exam_name: str, csv_path: str) -> str:
        """
        Create answer key from CSV file
        CSV should have columns: question, answer
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns")
        
        answers = {}
        for _, row in df.iterrows():
            question_num = int(row['question'])
            answer = str(row['answer']).upper()
            answers[question_num] = answer
        
        metadata = {
            'source': csv_path,
            'imported_from_csv': True
        }
        
        return self.create_answer_key(exam_name, answers, metadata)
    
    def export_to_csv(self, exam_name_or_path: str, output_path: str):
        """Export answer key to CSV format"""
        answer_key_data = self.load_answer_key(exam_name_or_path)
        answers = answer_key_data['answers']
        
        # Convert to list of dictionaries for DataFrame
        data = []
        for question_num in sorted(answers.keys(), key=int):
            data.append({
                'question': question_num,
                'answer': answers[str(question_num)] if isinstance(question_num, str) else answers[question_num]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Answer key exported to: {output_path}")
    
    def validate_answer_key(self, answers: Dict) -> Dict:
        """
        Validate an answer key for common issues
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check for missing questions
        if answers:
            question_numbers = set(int(k) if isinstance(k, str) else k for k in answers.keys())
            expected_questions = set(range(1, max(question_numbers) + 1))
            missing = expected_questions - question_numbers
            
            if missing:
                issues.append(f"Missing questions: {sorted(missing)}")
        
        # Check for invalid answers
        valid_options = {'A', 'B', 'C', 'D'}
        for q_num, answer in answers.items():
            if answer not in valid_options:
                issues.append(f"Question {q_num}: Invalid answer '{answer}' (should be A, B, C, or D)")
        
        # Check for common patterns that might be mistakes
        answer_distribution = {}
        for answer in answers.values():
            answer_distribution[answer] = answer_distribution.get(answer, 0) + 1
        
        total_questions = len(answers)
        for option, count in answer_distribution.items():
            percentage = (count / total_questions) * 100
            if percentage > 60:  # More than 60% of one answer is suspicious
                warnings.append(f"Answer '{option}' appears {percentage:.1f}% of the time ({count}/{total_questions})")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'answer_distribution': answer_distribution
        }
    
    def compare_answer_keys(self, key1_path: str, key2_path: str) -> Dict:
        """Compare two answer keys"""
        key1 = self.load_answer_key(key1_path)
        key2 = self.load_answer_key(key2_path)
        
        answers1 = key1['answers']
        answers2 = key2['answers']
        
        all_questions = set(answers1.keys()) | set(answers2.keys())
        differences = []
        matches = []
        
        for q in sorted(all_questions, key=int):
            q_str = str(q)
            answer1 = answers1.get(q_str, answers1.get(int(q), 'MISSING'))
            answer2 = answers2.get(q_str, answers2.get(int(q), 'MISSING'))
            
            if answer1 != answer2:
                differences.append({
                    'question': q,
                    'key1_answer': answer1,
                    'key2_answer': answer2
                })
            else:
                matches.append(q)
        
        return {
            'total_questions': len(all_questions),
            'matches': len(matches),
            'differences': len(differences),
            'match_percentage': (len(matches) / len(all_questions)) * 100 if all_questions else 0,
            'difference_details': differences
        }

def create_sample_answer_keys():
    """Create some sample answer keys for testing"""
    manager = AnswerKeyManager()
    
    # Sample Answer Key 1 - Mixed distribution
    sample_answers_1 = {
        1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A',
        6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',
        11: 'C', 12: 'D', 13: 'A', 14: 'B', 15: 'C',
        16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'D',
        21: 'A', 22: 'B', 23: 'C', 24: 'D', 25: 'A',
        26: 'B', 27: 'C', 28: 'D', 29: 'A', 30: 'B'
    }
    
    metadata_1 = {
        'subject': 'Mathematics',
        'exam_date': '2025-09-28',
        'difficulty': 'Medium',
        'description': 'Sample math exam with balanced answer distribution'
    }
    
    manager.create_answer_key("Sample_Math_Exam", sample_answers_1, metadata_1)
    
    # Sample Answer Key 2 - Different pattern
    sample_answers_2 = {
        1: 'B', 2: 'A', 3: 'D', 4: 'C', 5: 'B',
        6: 'A', 7: 'D', 8: 'C', 9: 'B', 10: 'A',
        11: 'D', 12: 'C', 13: 'B', 14: 'A', 15: 'D',
        16: 'C', 17: 'B', 18: 'A', 19: 'D', 20: 'C',
        21: 'B', 22: 'A', 23: 'D', 24: 'C', 25: 'B',
        26: 'A', 27: 'D', 28: 'C', 29: 'B', 30: 'A'
    }
    
    metadata_2 = {
        'subject': 'English',
        'exam_date': '2025-09-28',
        'difficulty': 'Hard',
        'description': 'Sample English exam with reverse pattern distribution'
    }
    
    manager.create_answer_key("Sample_English_Exam", sample_answers_2, metadata_2)
    
    print("Sample answer keys created successfully!")
    
    # List all answer keys
    print("\\nAvailable Answer Keys:")
    keys = manager.list_answer_keys()
    for key_info in keys:
        print(f"- {key_info['exam_name']} ({key_info['total_questions']} questions)")

def main():
    """Demonstration of answer key management"""
    manager = AnswerKeyManager()
    
    # Create sample answer keys
    create_sample_answer_keys()
    
    # Validate an answer key
    print("\\n" + "="*50)
    print("Validating Sample Math Exam...")
    sample_key = manager.load_answer_key("Sample_Math_Exam")
    validation = manager.validate_answer_key(sample_key['answers'])
    
    print(f"Valid: {validation['is_valid']}")
    if validation['issues']:
        print("Issues:", validation['issues'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
    
    print("Answer distribution:", validation['answer_distribution'])
    
    # Compare answer keys
    print("\\n" + "="*50)
    print("Comparing Sample Math and English Exams...")
    comparison = manager.compare_answer_keys("Sample_Math_Exam", "Sample_English_Exam")
    print(f"Match percentage: {comparison['match_percentage']:.1f}%")
    print(f"Matches: {comparison['matches']}/{comparison['total_questions']}")
    
    if comparison['difference_details']:
        print("First 5 differences:")
        for diff in comparison['difference_details'][:5]:
            print(f"Q{diff['question']}: {diff['key1_answer']} vs {diff['key2_answer']}")

if __name__ == "__main__":
    main()