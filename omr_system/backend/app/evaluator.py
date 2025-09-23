from typing import Dict, List
import uuid
from datetime import datetime

class Evaluator:
    def __init__(self):
        pass
    
    def evaluate(self, omr_data: Dict, answer_key: Dict) -> Dict:
        """
        Evaluate OMR answers against answer key
        Returns detailed evaluation results
        """
        student_answers = omr_data.get('answers', {})
        correct_answers = answer_key.get('answers', {})
        
        # Initialize counters
        total_questions = answer_key.get('total_questions', len(correct_answers))
        correct_count = 0
        wrong_count = 0
        blank_count = 0
        
        # Detailed answer analysis
        question_analysis = {}
        
        for question_num in range(1, total_questions + 1):
            question_str = str(question_num)
            student_answer = student_answers.get(question_str, "").strip().upper()
            correct_answer = correct_answers.get(question_str, "").strip().upper()
            
            if not student_answer:
                # Blank answer
                blank_count += 1
                status = "BLANK"
            elif student_answer == correct_answer:
                # Correct answer
                correct_count += 1
                status = "CORRECT"
            else:
                # Wrong answer
                wrong_count += 1
                status = "WRONG"
            
            question_analysis[question_str] = {
                "student_answer": student_answer or "BLANK",
                "correct_answer": correct_answer,
                "status": status
            }
        
        # Calculate marks (you can customize this logic)
        marks_per_question = 1  # Default: 1 mark per question
        negative_marking = 0.25  # Default: -0.25 for wrong answers
        
        total_marks = (correct_count * marks_per_question) - (wrong_count * negative_marking)
        max_possible_marks = total_questions * marks_per_question
        
        # Calculate percentage
        percentage = (total_marks / max_possible_marks * 100) if max_possible_marks > 0 else 0
        
        # Prepare result
        result = {
            "evaluation_id": str(uuid.uuid4()),
            "roll_number": omr_data.get('roll_number'),
            "branch": omr_data.get('branch'),
            "class": omr_data.get('class'),
            "subject": omr_data.get('subject'),
            "exam_date": omr_data.get('exam_date'),
            "set_code": omr_data.get('set_code'),
            "answer_key_id": answer_key.get('answer_key_id'),
            "file_id": omr_data.get('file_id'),
            
            # Summary statistics
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "wrong_answers": wrong_count,
            "blank_answers": blank_count,
            "total_marks": round(total_marks, 2),
            "max_possible_marks": max_possible_marks,
            "percentage": round(percentage, 2),
            
            # Detailed analysis
            "question_analysis": question_analysis,
            
            # Metadata
            "evaluation_time": datetime.now().isoformat(),
            "marks_per_question": marks_per_question,
            "negative_marking": negative_marking
        }
        
        return result
    
    def batch_evaluate(self, omr_data_list: List[Dict], answer_key: Dict) -> List[Dict]:
        """Evaluate multiple OMR sheets against the same answer key"""
        results = []
        
        for omr_data in omr_data_list:
            result = self.evaluate(omr_data, answer_key)
            results.append(result)
        
        return results
    
    def generate_class_report(self, results: List[Dict]) -> Dict:
        """Generate class-wise performance report"""
        if not results:
            return {}
        
        total_students = len(results)
        total_marks_sum = sum(result['total_marks'] for result in results)
        percentage_sum = sum(result['percentage'] for result in results)
        
        # Calculate statistics
        average_marks = total_marks_sum / total_students
        average_percentage = percentage_sum / total_students
        
        # Find highest and lowest scores
        highest_score = max(results, key=lambda x: x['total_marks'])
        lowest_score = min(results, key=lambda x: x['total_marks'])
        
        # Grade distribution (you can customize grade boundaries)
        grade_distribution = {
            "A+": len([r for r in results if r['percentage'] >= 90]),
            "A": len([r for r in results if 80 <= r['percentage'] < 90]),
            "B": len([r for r in results if 70 <= r['percentage'] < 80]),
            "C": len([r for r in results if 60 <= r['percentage'] < 70]),
            "D": len([r for r in results if 50 <= r['percentage'] < 60]),
            "F": len([r for r in results if r['percentage'] < 50])
        }
        
        # Question-wise analysis
        if results and 'question_analysis' in results[0]:
            total_questions = len(results[0]['question_analysis'])
            question_stats = {}
            
            for q_num in range(1, total_questions + 1):
                q_str = str(q_num)
                correct = sum(1 for r in results if r['question_analysis'].get(q_str, {}).get('status') == 'CORRECT')
                wrong = sum(1 for r in results if r['question_analysis'].get(q_str, {}).get('status') == 'WRONG')
                blank = sum(1 for r in results if r['question_analysis'].get(q_str, {}).get('status') == 'BLANK')
                
                question_stats[q_str] = {
                    "correct": correct,
                    "wrong": wrong,
                    "blank": blank,
                    "difficulty": "EASY" if (correct / total_students) > 0.8 else "MEDIUM" if (correct / total_students) > 0.5 else "HARD"
                }
        
        return {
            "report_id": str(uuid.uuid4()),
            "generated_time": datetime.now().isoformat(),
            "total_students": total_students,
            "average_marks": round(average_marks, 2),
            "average_percentage": round(average_percentage, 2),
            "highest_score": {
                "roll_number": highest_score['roll_number'],
                "marks": highest_score['total_marks'],
                "percentage": highest_score['percentage']
            },
            "lowest_score": {
                "roll_number": lowest_score['roll_number'],
                "marks": lowest_score['total_marks'],
                "percentage": lowest_score['percentage']
            },
            "grade_distribution": grade_distribution,
            "question_analysis": question_stats if 'question_stats' in locals() else {}
        }