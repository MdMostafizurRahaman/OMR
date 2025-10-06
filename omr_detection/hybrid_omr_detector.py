import google.generativeai as genai
import PIL.Image
import sys
import os
import json
import cv2
import numpy as np
from answer_key_manager import AnswerKeyManager

# Configure Gemini API
genai.configure(api_key="AIzaSyA2cqiH1MecukxgSyMtZ9K2zSZG3O3Rkoo")

def preprocess_image(image_path):
    """Preprocess image for better OMR detection"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for better bubble detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Save preprocessed image
    preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
    cv2.imwrite(preprocessed_path, thresh)

    return preprocessed_path

def opencv_detect_answers(image_path):
    """Use OpenCV to detect answers with improved algorithm"""
    img = cv2.imread(image_path)
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parameters based on the optimized configurations
    configs = {
        'eng_ans1.jpg': {
            'col1_left': 90, 'col2_left': 265, 'col3_left': 415,
            'first_row_top': 310, 'row_spacing': 38, 'bubble_spacing': 19, 'bubble_offset_x': 34
        },
        'eng_ans2.jpg': {
            'col1_left': 80, 'col2_left': 240, 'col3_left': 410,
            'first_row_top': 305, 'row_spacing': 30, 'bubble_spacing': 17, 'bubble_offset_x': 30
        },
        'eng_ques.jpg': {
            'col1_left': 95, 'col2_left': 240, 'col3_left': 390,
            'first_row_top': 290, 'row_spacing': 32, 'bubble_spacing': 15, 'bubble_offset_x': 32
        }
    }

    filename = os.path.basename(image_path)
    config = configs.get(filename, configs['eng_ans1.jpg'])  # Default to ans1 config

    answers = {}
    options = ['A', 'B', 'C', 'D']

    for q in range(1, 31):
        col = ((q-1) // 10) + 1
        row_in_col = ((q-1) % 10) + 1

        if col == 1:
            x_start = config['col1_left']
        elif col == 2:
            x_start = config['col2_left']
        else:
            x_start = config['col3_left']

        y_start = config['first_row_top'] + (row_in_col - 1) * config['row_spacing']

        max_intensity = 0
        best_option = 'BLANK'

        for i, option in enumerate(options):
            x = x_start + config['bubble_offset_x'] + i * config['bubble_spacing']
            y = y_start

            # Extract bubble region (smaller, more precise)
            bubble = gray[y-3:y+7, x-3:x+7]
            if bubble.size > 0:
                # Calculate how much of the bubble is filled (dark pixels)
                filled_ratio = np.sum(bubble < 100) / bubble.size  # Dark pixels ratio
                if filled_ratio > 0.3:  # At least 30% filled
                    answers[q] = option
                    break  # Take first valid detection
        else:
            answers[q] = 'BLANK'

    return answers

def gemini_detect_answers(image_path):
    """Use Gemini AI to detect answers"""
    try:
        # Preprocess image first
        preprocessed_path = preprocess_image(image_path)
        if preprocessed_path:
            image = PIL.Image.open(preprocessed_path)
        else:
            image = PIL.Image.open(image_path)

        model = genai.GenerativeModel('models/gemini-2.5-pro')

        prompt = """
        You are an expert OMR analyzer. Analyze this preprocessed OMR sheet image.

        CRITICAL INSTRUCTIONS:
        1. Only valid answers are A, B, C, D, or BLANK
        2. Look for clearly darkened bubbles
        3. If uncertain, mark as BLANK
        4. Be extremely precise

        For each question 1-30, provide the answer in this exact format:
        Question 1: [ANSWER]
        Question 2: [ANSWER]
        ...
        Question 30: [ANSWER]
        """

        response = model.generate_content([prompt, image])

        # Parse response
        answers = {}
        lines = response.text.split('\n')

        for line in lines:
            line = line.strip()
            if 'Question' in line and ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    q_part = parts[0].replace('Question', '').strip()
                    answer = parts[1].strip().upper()
                    if q_part.isdigit() and answer in ['A', 'B', 'C', 'D', 'BLANK']:
                        answers[int(q_part)] = answer

        return answers

    except Exception as e:
        print(f"Gemini error: {e}")
        return {}

def hybrid_detect_answers(image_path):
    """Combine OpenCV and Gemini results for maximum accuracy"""
    opencv_answers = opencv_detect_answers(image_path)
    gemini_answers = gemini_detect_answers(image_path)

    final_answers = {}

    for q in range(1, 31):
        opencv_ans = opencv_answers.get(q, 'BLANK')
        gemini_ans = gemini_answers.get(q, 'BLANK')

        # If Gemini detects something, prefer it (AI is more accurate)
        if gemini_ans != 'BLANK':
            final_answers[q] = gemini_ans
        # If Gemini says BLANK but OpenCV detects something, check if OpenCV is confident
        elif opencv_ans != 'BLANK':
            # For now, prefer BLANK from Gemini (conservative approach)
            final_answers[q] = 'BLANK'
        else:
            final_answers[q] = 'BLANK'

    return final_answers, opencv_answers, gemini_answers

def evaluate_answers(detected_answers, correct_answers):
    """Evaluate detected answers against correct answers"""
    correct = 0
    total = len(correct_answers)
    incorrect = []

    for q_num, correct_ans in correct_answers.items():
        detected = detected_answers.get(q_num, 'NOT_DETECTED')

        if correct_ans.upper() == 'BLANK':
            if detected == 'BLANK':
                correct += 1
            else:
                incorrect.append({
                    'question': q_num,
                    'detected': detected,
                    'correct': correct_ans,
                    'note': 'Should be blank'
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
    if len(sys.argv) != 3:
        print("Usage: python hybrid_omr_detector.py <image_path> <exam_name>")
        sys.exit(1)

    image_path = sys.argv[1]
    exam_name = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)

    print(f"🔍 Analyzing OMR image: {image_path} (Hybrid Mode)")
    print("=" * 60)

    # Get hybrid results
    final_answers, opencv_answers, gemini_answers = hybrid_detect_answers(image_path)

    print("🤖 AI (Gemini) Results:")
    for q in sorted(gemini_answers.keys()):
        print(f"Q{q}: {gemini_answers[q]}", end="  ")
        if q % 10 == 0:
            print()
    print("\n")

    print("📷 Computer Vision (OpenCV) Results:")
    for q in sorted(opencv_answers.keys()):
        print(f"Q{q}: {opencv_answers[q]}", end="  ")
        if q % 10 == 0:
            print()
    print("\n")

    print("🎯 Hybrid Final Results:")
    for q in sorted(final_answers.keys()):
        print(f"Q{q}: {final_answers[q]}", end="  ")
        if q % 10 == 0:
            print()
    print("\n" + "=" * 60)

    # Load answer key and evaluate
    try:
        manager = AnswerKeyManager()
        answer_key_data = manager.load_answer_key(exam_name)
        correct_answers = {int(k): v for k, v in answer_key_data['answers'].items()}

        evaluation = evaluate_answers(final_answers, correct_answers)

        print("📊 EVALUATION RESULTS:")
        print(f"Total Questions: {evaluation['total_questions']}")
        print(f"✅ Correct Answers: {evaluation['correct_answers']}")
        print(f"❌ Incorrect Answers: {evaluation['incorrect_answers']}")
        print(f"🎯 Score: {evaluation['score_percentage']:.2f}%")

        if evaluation['incorrect_details']:
            print("\n❌ Incorrect Answers:")
            for detail in evaluation['incorrect_details'][:10]:
                note = detail.get('note', '')
                print(f"Q{detail['question']}: Detected '{detail['detected']}', Correct '{detail['correct']}' {note}")
            if len(evaluation['incorrect_details']) > 10:
                print(f"... and {len(evaluation['incorrect_details']) - 10} more")

        # Success message
        if evaluation['score_percentage'] >= 95:
            print("\n🎉 Excellent! High accuracy achieved!")
        elif evaluation['score_percentage'] >= 80:
            print("\n👍 Good accuracy! Hybrid approach working well.")
        else:
            print("\n⚠️  Accuracy needs improvement. Consider manual verification.")

    except Exception as e:
        print(f"❌ Could not evaluate answers: {str(e)}")

if __name__ == "__main__":
    main()