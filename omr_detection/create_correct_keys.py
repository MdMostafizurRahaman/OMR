from answer_key_manager import AnswerKeyManager

def create_correct_answer_keys():
    """Create answer keys with the correct answers provided by the user"""

    manager = AnswerKeyManager()

    # Correct answers for eng_ans1.jpg
    eng_ans1_answers = {
        1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'B', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'D',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'A', 28: 'B', 29: 'C', 30: 'D'
    }

    metadata_ans1 = {
        'roll_number': '132013',
        'set_code': 'A',
        'description': 'Correct answers for eng_ans1.jpg OMR sheet',
        'source': 'User provided correct answers'
    }

    manager.create_answer_key("eng_ans1_correct", eng_ans1_answers, metadata_ans1)

    # Correct answers for eng_ans2.jpg
    eng_ans2_answers = {
        1: 'D', 2: 'C', 3: 'A', 4: 'D', 5: 'BLANK', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'C', 15: 'C', 16: 'D', 17: 'D', 18: 'B', 19: 'A', 20: 'BLANK',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'BLANK', 28: 'BLANK', 29: 'BLANK', 30: 'BLANK'
    }

    metadata_ans2 = {
        'roll_number': '132713',
        'set_code': 'A',
        'description': 'Correct answers for eng_ans2.jpg OMR sheet',
        'source': 'User provided correct answers'
    }

    manager.create_answer_key("eng_ans2_correct", eng_ans2_answers, metadata_ans2)

    # Correct answers for eng_ques.jpg
    eng_ques_answers = {
        1: 'D', 2: 'C', 3: 'A', 4: 'A', 5: 'C', 6: 'C', 7: 'B', 8: 'A', 9: 'D', 10: 'A',
        11: 'A', 12: 'D', 13: 'B', 14: 'D', 15: 'C', 16: 'A', 17: 'D', 18: 'B', 19: 'A', 20: 'A',
        21: 'A', 22: 'C', 23: 'C', 24: 'C', 25: 'A', 26: 'D', 27: 'B', 28: 'D', 29: 'A', 30: 'B'
    }

    metadata_ques = {
        'roll_number': '000000',
        'set_code': 'A',
        'description': 'Correct answers for eng_ques.jpg OMR sheet',
        'source': 'User provided correct answers'
    }

    manager.create_answer_key("eng_ques_correct", eng_ques_answers, metadata_ques)

    print("Correct answer keys created successfully!")
    print("\nAvailable Answer Keys:")
    keys = manager.list_answer_keys()
    for key_info in keys:
        if 'correct' in key_info['exam_name']:
            print(f"- {key_info['exam_name']} ({key_info['total_questions']} questions)")

if __name__ == "__main__":
    create_correct_answer_keys()