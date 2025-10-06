import google.generativeai as genai
import PIL.Image
import sys
import os
import json
from answer_key_manager import AnswerKeyManager

# Configure Gemini API
genai.configure(api_key="AIzaSyA2cqiH1MecukxgSyMtZ9K2zSZG3O3Rkoo")

def analyze_omr_image(image_path, num_calls=3):
    """
    Analyze OMR image using multiple Gemini calls for consensus
    """
    try:
        image = PIL.Image.open(image_path)
        
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        
        prompt = """
        You are an expert OMR (Optical Mark Recognition) analyzer. Your task is to accurately identify which bubbles are filled on this OMR sheet.

        CRITICAL INSTRUCTIONS:
        1. Only report answers for questions that have CLEARLY FILLED bubbles
        2. Valid options are ONLY: A, B, C, D, or BLANK
        3. If a bubble is partially filled, smudged, or unclear, mark it as BLANK
        4. Look carefully at each question's bubble row
        5. Be extremely precise - this is for important examination evaluation

        Analyze questions 1-30. For each question, determine if there is a clearly marked bubble (A, B, C, or D) or if it should be BLANK.

        Format your response exactly like this:
        Question 1: [A/B/C/D/BLANK]
        Question 2: [A/B/C/D/BLANK]
        ...
        Question 30: [A/B/C/D/BLANK]

        Do not include any other text or explanations.
        """
        
        # Get multiple responses
        responses = []
        for i in range(num_calls):
            try:
                response = model.generate_content([prompt, image])
                responses.append(response.text)
            except:
                continue
        
        if not responses:
            return "Error: No responses from AI"
        
        # Parse all responses and find consensus
        all_answers = []
        for response in responses:
            answers = parse_answers_from_response(response)
            all_answers.append(answers)
        
        # Take consensus
        final_answers = {}
        for q in range(1, 31):
            votes = {}
            for answers in all_answers:
                ans = answers.get(q, 'BLANK')
                votes[ans] = votes.get(ans, 0) + 1
            
            # Take majority vote
            best_ans = max(votes.items(), key=lambda x: x[1])
            final_answers[q] = best_ans[0]
        
        # Format as text
        result = f"Consensus Analysis (from {len(responses)} AI calls):\n"
        for q in range(1, 31):
            result += f"Question {q}: {final_answers[q]}\n"
        
        return result
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def parse_answers_from_response(response_text):
    """
    Parse the Gemini response to extract answers as a dictionary.
    """
    answers = {}
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for patterns like "*   **Question 1:** B" or "**Question 1: B**"
        if 'Question' in line and ':' in line:
            try:
                # Remove markdown
                clean_line = line.replace('*', '').replace('**', '').strip()
                if 'Question' in clean_line:
                    parts = clean_line.split(':')
                    if len(parts) >= 2:
                        question_part = parts[0].replace('Question', '').strip()
                        answer = parts[1].strip()
                        if question_part.isdigit():
                            answers[int(question_part)] = answer.upper()
            except:
                continue
    
    return answers

def evaluate_answers(detected_answers, correct_answers):
    """
    Evaluate the detected answers against correct answers.
    Handles BLANK answers specially.
    """
    correct = 0
    total = len(correct_answers)
    incorrect = []
    
    for q_num, correct_ans in correct_answers.items():
        detected = detected_answers.get(q_num, 'NOT_DETECTED')
        
        # Handle blank answers
        if correct_ans.upper() == 'BLANK':
            if detected == 'NOT_DETECTED':
                correct += 1
            else:
                incorrect.append({
                    'question': q_num,
                    'detected': detected,
                    'correct': correct_ans,
                    'note': 'Should be blank but detected answer'
                })
        else:
            if detected == correct_ans.upper():
                correct += 1
            else:
                incorrect.append({
                    'question': q_num,
                    'detected': detected,
                    'correct': correct_ans
                })
    
    score_percentage = (correct / total) * 100 if total > 0 else 0
    
    return {
        'total_questions': total,
        'correct_answers': correct,
        'incorrect_answers': len(incorrect),
        'score_percentage': score_percentage,
        'incorrect_details': incorrect
    }

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python gemini_omr_detector.py <image_path> <exam_name> [num_calls]")
        print("Example: python gemini_omr_detector.py eng_ans1.jpg eng_ans1_correct 5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    exam_name = sys.argv[2]
    num_calls = int(sys.argv[3]) if len(sys.argv) == 4 else 3
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing OMR image: {image_path} (Consensus from {num_calls} AI calls)")
    print("=" * 50)
    
    # Analyze the image
    response = analyze_omr_image(image_path, num_calls)
    print("Gemini Analysis:")
    print(response)
    print("=" * 50)
    
    # Parse answers
    detected_answers = parse_answers_from_response(response)
    print(f"Detected {len(detected_answers)} answers:")
    for q, ans in sorted(detected_answers.items()):
        print(f"Question {q}: {ans}", end="  ")
        if q % 10 == 0:
            print()
    print("\n" + "=" * 50)
    
    # Load answer key and evaluate
    try:
        manager = AnswerKeyManager()
        answer_key_data = manager.load_answer_key(exam_name)
        correct_answers = answer_key_data['answers']
        
        # Convert string keys to int if needed
        correct_answers = {int(k): v for k, v in correct_answers.items()}
        
        evaluation = evaluate_answers(detected_answers, correct_answers)
        
        print("EVALUATION RESULTS:")
        print(f"Total Questions: {evaluation['total_questions']}")
        print(f"Correct Answers: {evaluation['correct_answers']}")
        print(f"Incorrect Answers: {evaluation['incorrect_answers']}")
        print(f"Score: {evaluation['score_percentage']:.2f}%")
        
        if evaluation['incorrect_details']:
            print("\nIncorrect Answers:")
            for detail in evaluation['incorrect_details'][:10]:  # Show first 10
                print(f"Q{detail['question']}: Detected '{detail['detected']}', Correct '{detail['correct']}'")
            if len(evaluation['incorrect_details']) > 10:
                print(f"... and {len(evaluation['incorrect_details']) - 10} more")
    
    except Exception as e:
        print(f"Could not evaluate answers: {str(e)}")
        print("Make sure the answer key exists. Use answer_key_manager.py to create one.")

if __name__ == "__main__":
    main()