"""
Create answer keys for the test images based on provided correct answers
"""

from answer_key_manager import AnswerKeyManager

def create_test_answer_keys():
    """Create answer keys for the test images"""
    manager = AnswerKeyManager()
    
    # Answer key for eng_ans1.jpg
    # Roll: 132013, set_code: A
    # Q1: A, Q2: B, Q3: B, Q4: C, Q5: blank, Q6: C, Q7: B, Q8: A, Q9: D, Q10: A, 
    # Q11: A, Q12: D, Q13: B, Q14: B, Q15: C, Q16: D, Q17: D, Q18: B, Q19: A, Q20: D, 
    # Q21: A, Q22: C Q23: C, Q24: C, Q25: A, Q26: D, Q27: A, Q28: B, Q29: C, Q30: D
    
    eng_ans1_answers = {
        1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
    }
    
    eng_ans1_metadata = {
        'roll_number': '132013',
        'set_code': 'A',
        'image_file': 'eng_ans1.jpg',
        'description': 'Answer key for eng_ans1.jpg - Student roll 132013'
    }
    
    manager.create_answer_key("eng_ans1", eng_ans1_answers, eng_ans1_metadata)
    
    # Answer key for eng_ans2.jpg
    # roll: 132713, set_code: A
    # Q1: D, Q2: C, Q3: A, Q4: D, Q5: blank, Q6: C, Q7: B, Q8: A, Q9: D, Q10: A, 
    # Q11: A, Q12: D, Q13: B, Q14: C, Q15: C, Q16: D, Q17: D, Q18: B, Q19: A, Q20: blank, 
    # Q21: A, Q22: C Q23: C, Q24: C, Q25: A, Q26: D, Q27: BLANK, Q28: BLANK, Q29: BLANK, Q30: BLANK
    
    eng_ans2_answers = {
        1: 'D', 2: 'C', 3: 'A', 4: 'D', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'C', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'BLANK',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'BLANK', 28: 'BLANK', 29: 'BLANK', 30: 'BLANK'
    }
    
    eng_ans2_metadata = {
        'roll_number': '132713',
        'set_code': 'A',
        'image_file': 'eng_ans2.jpg',
        'description': 'Answer key for eng_ans2.jpg - Student roll 132713'
    }
    
    manager.create_answer_key("eng_ans2", eng_ans2_answers, eng_ans2_metadata)
    
    # Answer key for eng_ques.jpg
    # roll: 000000, set_code: A
    # Q1: D, Q2: C, Q3: A, Q4: A, Q5: C, Q6: C, Q7: B, Q8: A, Q9: D, Q10: A, 
    # Q11: A, Q12: D, Q13: B, Q14: D, Q15: C, Q16: A, Q17: D, Q18: B, Q19: A, Q20: A, 
    # Q21: A, Q22: C Q23: C, Q24: C, Q25: A, Q26: D, Q27: B, Q28: D, Q29: A, Q30: B
    
    eng_ques_answers = {
        1: 'D', 2: 'C', 3: 'A', 4: 'A', 5: 'C', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'D', 15: 'C', 16: 'A', 17: 'D', 18: 'B', 19: 'A', 20: 'A',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'B', 28: 'D', 29: 'A', 30: 'B'
    }
    
    eng_ques_metadata = {
        'roll_number': '000000',
        'set_code': 'A',
        'image_file': 'eng_ques.jpg',
        'description': 'Answer key for eng_ques.jpg - Template/question paper roll 000000'
    }
    
    manager.create_answer_key("eng_ques", eng_ques_answers, eng_ques_metadata)
    
    print("All test answer keys created successfully!")
    
    # List all answer keys
    print("\nAvailable Answer Keys:")
    keys = manager.list_answer_keys()
    for key_info in keys:
        print(f"- {key_info['exam_name']} ({key_info['total_questions']} questions)")
        
    # Validate the answer keys
    for key_name in ["eng_ans1", "eng_ans2", "eng_ques"]:
        print(f"\nValidating {key_name}...")
        key_data = manager.load_answer_key(key_name)
        validation = manager.validate_answer_key(key_data['answers'])
        
        print(f"  Valid: {validation['is_valid']}")
        if validation['issues']:
            print(f"  Issues: {validation['issues']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        
        # Count blank answers
        blank_count = sum(1 for a in key_data['answers'].values() if a == 'BLANK')
        filled_count = sum(1 for a in key_data['answers'].values() if a != 'BLANK')
        print(f"  Filled: {filled_count}, Blank: {blank_count}")

if __name__ == "__main__":
    create_test_answer_keys()