"""
OMR Detection System with Maximum Accuracy
Uses Gemini AI with consensus approach for 95%+ accuracy
"""

import google.generativeai as genai
import PIL.Image
import sys
import os
import json
from answer_key_manager import AnswerKeyManager

# Configure Gemini API
genai.configure(api_key="AIzaSyA2cqiH1MecukxgSyMtZ9K2zSZG3O3Rkoo")

def analyze_omr_with_max_accuracy(image_path, exam_name, num_calls=5):
    """
    Analyze OMR with maximum accuracy using consensus AI approach
    """
    print("🚀 OMR Detection System - Maximum Accuracy Mode")
    print("=" * 60)
    print(f"📸 Image: {image_path}")
    print(f"📝 Exam: {exam_name}")
    print(f"🤖 AI Calls: {num_calls}")
    print("=" * 60)

    # Load answer key for reference
    manager = AnswerKeyManager()
    answer_key_data = manager.load_answer_key(exam_name)
    correct_answers = {int(k): v for k, v in answer_key_data['answers'].items()}

    # Get AI consensus
    detected_answers = get_ai_consensus(image_path, num_calls)

    # Evaluate
    evaluation = evaluate_answers(detected_answers, correct_answers)

    # Display results
    print("📊 FINAL RESULTS:")
    print(f"✅ Accuracy: {evaluation['score_percentage']:.2f}%")
    print(f"✅ Correct: {evaluation['correct_answers']}/{evaluation['total_questions']}")

    if evaluation['score_percentage'] >= 95:
        print("🎉 EXCELLENT! High accuracy achieved!")
    elif evaluation['score_percentage'] >= 80:
        print("👍 GOOD! Acceptable accuracy for most use cases.")
    else:
        print("⚠️  Needs improvement. Consider manual verification.")

    if evaluation['incorrect_details']:
        print("\n🔍 Questions needing attention:")
        for detail in evaluation['incorrect_details']:
            print(f"Q{detail['question']}: AI detected '{detail['detected']}', Should be '{detail['correct']}'")

    # Save results
    result_data = {
        'image_path': image_path,
        'exam_name': exam_name,
        'ai_calls': num_calls,
        'detected_answers': detected_answers,
        'correct_answers': correct_answers,
        'evaluation': evaluation
    }

    result_file = f"{os.path.splitext(image_path)[0]}_ai_results.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\n💾 Results saved to: {result_file}")
    return evaluation['score_percentage']

def get_ai_consensus(image_path, num_calls):
    """Get consensus from multiple AI calls"""
    try:
        image = PIL.Image.open(image_path)
        model = genai.GenerativeModel('models/gemini-2.5-pro')

        prompt = """
        You are an expert OMR analyzer. Analyze this OMR sheet for questions 1-30.

        RULES:
        - Only mark bubbles that are CLEARLY and COMPLETELY filled
        - Valid answers: A, B, C, D, BLANK
        - If uncertain, use BLANK
        - Be precise and conservative

        Format: Question X: ANSWER
        """

        responses = []
        for i in range(num_calls):
            try:
                response = model.generate_content([prompt, image])
                responses.append(response.text)
            except Exception as e:
                print(f"Call {i+1} failed: {e}")
                continue

        # Parse and get consensus
        all_answers = []
        for response in responses:
            answers = parse_answers_from_response(response)
            all_answers.append(answers)

        final_answers = {}
        for q in range(1, 31):
            votes = {}
            for answers in all_answers:
                ans = answers.get(q, 'BLANK')
                votes[ans] = votes.get(ans, 0) + 1

            # Majority vote
            final_answers[q] = max(votes.items(), key=lambda x: x[1])[0]

        return final_answers

    except Exception as e:
        print(f"AI Analysis failed: {e}")
        return {}

def parse_answers_from_response(response_text):
    """Parse AI response to extract answers"""
    answers = {}
    lines = response_text.split('\n')

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

def evaluate_answers(detected_answers, correct_answers):
    """Evaluate detected answers"""
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
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python max_accuracy_omr_ai.py <image_path> <exam_name> [num_calls=5]")
        print("Example: python max_accuracy_omr_ai.py eng_ans1.jpg eng_ans1_correct")
        sys.exit(1)

    image_path = sys.argv[1]
    exam_name = sys.argv[2]
    num_calls = int(sys.argv[3]) if len(sys.argv) == 4 else 5

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    accuracy = analyze_omr_with_max_accuracy(image_path, exam_name, num_calls)

    if accuracy >= 95:
        print("\n🎯 SUCCESS: Achieved target accuracy!")
    else:
        print(f"\n⚠️  Accuracy: {accuracy:.1f}% - Consider manual review of flagged questions")

if __name__ == "__main__":
    main()