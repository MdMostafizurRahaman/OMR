"""
Main OMR Processing Script
High-accuracy OMR detection and evaluation system
"""

import os
import sys
import argparse
from precise_omr_detector import PreciseOMRDetector
from answer_key_manager import AnswerKeyManager
from omr_calibrator import OMRCalibrator

def main():
    parser = argparse.ArgumentParser(description='OMR Detection and Evaluation System')
    parser.add_argument('mode', choices=['detect', 'calibrate', 'manage_keys'], 
                       help='Operation mode')
    parser.add_argument('--image', '-i', help='Path to OMR image file')
    parser.add_argument('--answer_key', '-a', help='Answer key name or path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--threshold', '-t', type=float, default=0.65,
                       help='Bubble detection threshold (0.3-0.9)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.mode == 'detect':
        detect_omr(args)
    elif args.mode == 'calibrate':
        run_calibration()
    elif args.mode == 'manage_keys':
        manage_answer_keys()

def detect_omr(args):
    """Run OMR detection on image(s)"""
    if not args.image:
        print("Error: Image path required for detection mode")
        return
    
    # Initialize detector
    detector = PreciseOMRDetector()
    detector.debug = args.debug
    detector.bubble_threshold = args.threshold
    
    # Load answer key if provided
    answer_key = None
    if args.answer_key:
        try:
            manager = AnswerKeyManager()
            key_data = manager.load_answer_key(args.answer_key)
            answer_key = key_data['answers']
            print(f"Loaded answer key: {key_data['exam_name']}")
        except Exception as e:
            print(f"Warning: Could not load answer key: {e}")
    
    # Process image
    try:
        results = detector.extract_answers(args.image, answer_key)
        
        # Display results
        print(f"\\n{'='*60}")
        print(f"OMR DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Image: {args.image}")
        print(f"Threshold: {args.threshold}")
        print(f"{'='*60}")
        
        # Show answers
        answers = results['answers']
        print("\\nDetected Answers:")
        print("-" * 30)
        
        for col in range(3):  # 3 columns
            print(f"\\nColumn {col+1}:")
            for row in range(10):  # 10 questions per column
                q_num = col * 10 + row + 1
                answer = answers.get(q_num, 'BLANK')
                print(f"Q{q_num:2d}: {answer}")
        
        # Show evaluation if answer key was provided
        if results['evaluation']:
            eval_summary = results['evaluation']['summary']
            print(f"\\n{'='*30}")
            print("EVALUATION SUMMARY")
            print(f"{'='*30}")
            print(f"Correct:   {eval_summary['correct']:2d}")
            print(f"Wrong:     {eval_summary['wrong']:2d}")
            print(f"Blank:     {eval_summary['blank']:2d}")
            print(f"Multiple:  {eval_summary['multiple']:2d}")
            print(f"Total:     {eval_summary['total']:2d}")
            print(f"Score:     {eval_summary['score']:5.1f}%")
        
        # Save results if output path provided
        if args.output:
            detector.save_results(results, args.output)
        
        # Show bubble analysis summary
        bubble_analysis = results['bubble_analysis']
        filled_bubbles = [b for b in bubble_analysis if b['is_filled']]
        
        print(f"\\n{'='*30}")
        print("DETECTION STATISTICS")
        print(f"{'='*30}")
        print(f"Total bubbles analyzed: {len(bubble_analysis)}")
        print(f"Bubbles detected as filled: {len(filled_bubbles)}")
        print(f"Average fill percentage of filled bubbles: {sum(b['fill_percentage'] for b in filled_bubbles) / len(filled_bubbles):.3f}" if filled_bubbles else "N/A")
        
        if args.debug:
            print(f"\\nDebug images saved to: debug_output/")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def run_calibration():
    """Run the calibration tool"""
    print("Starting OMR Calibration Tool...")
    print("This will open a GUI for interactive calibration.")
    
    try:
        calibrator = OMRCalibrator()
        calibrator.run()
    except Exception as e:
        print(f"Error starting calibration tool: {e}")
        print("Make sure you have tkinter installed (usually comes with Python)")

def manage_answer_keys():
    """Interactive answer key management"""
    manager = AnswerKeyManager()
    
    while True:
        print(f"\\n{'='*50}")
        print("ANSWER KEY MANAGEMENT")
        print(f"{'='*50}")
        print("1. List existing answer keys")
        print("2. Create new answer key")
        print("3. Create answer key from CSV")
        print("4. Export answer key to CSV")
        print("5. Validate answer key")
        print("6. Compare two answer keys")
        print("7. Create sample answer keys")
        print("8. Exit")
        
        choice = input("\\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            keys = manager.list_answer_keys()
            if keys:
                print("\\nExisting Answer Keys:")
                for i, key in enumerate(keys, 1):
                    print(f"{i}. {key['exam_name']} ({key['total_questions']} questions)")
            else:
                print("No answer keys found.")
        
        elif choice == '2':
            exam_name = input("Enter exam name: ").strip()
            total_q = int(input("Total questions (default 30): ") or 30)
            
            print("Enter answers for each question (A, B, C, D):")
            answers = {}
            for q in range(1, total_q + 1):
                while True:
                    answer = input(f"Q{q}: ").strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        answers[q] = answer
                        break
                    else:
                        print("Please enter A, B, C, or D")
            
            manager.create_answer_key(exam_name, answers)
            print("Answer key created successfully!")
        
        elif choice == '3':
            csv_path = input("Enter CSV file path: ").strip()
            exam_name = input("Enter exam name: ").strip()
            try:
                manager.create_from_csv(exam_name, csv_path)
                print("Answer key created from CSV successfully!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            exam_name = input("Enter exam name or path: ").strip()
            output_path = input("Enter output CSV path: ").strip()
            try:
                manager.export_to_csv(exam_name, output_path)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            exam_name = input("Enter exam name or path: ").strip()
            try:
                key_data = manager.load_answer_key(exam_name)
                validation = manager.validate_answer_key(key_data['answers'])
                
                print(f"\\nValidation Results:")
                print(f"Valid: {validation['is_valid']}")
                if validation['issues']:
                    print("Issues:")
                    for issue in validation['issues']:
                        print(f"  - {issue}")
                if validation['warnings']:
                    print("Warnings:")
                    for warning in validation['warnings']:
                        print(f"  - {warning}")
                print(f"Answer distribution: {validation['answer_distribution']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '6':
            key1 = input("Enter first answer key name/path: ").strip()
            key2 = input("Enter second answer key name/path: ").strip()
            try:
                comparison = manager.compare_answer_keys(key1, key2)
                print(f"\\nComparison Results:")
                print(f"Match percentage: {comparison['match_percentage']:.1f}%")
                print(f"Matches: {comparison['matches']}/{comparison['total_questions']}")
                
                if comparison['difference_details']:
                    print("\\nDifferences:")
                    for diff in comparison['difference_details'][:10]:  # Show first 10
                        print(f"Q{diff['question']}: {diff['key1_answer']} vs {diff['key2_answer']}")
                    if len(comparison['difference_details']) > 10:
                        print(f"... and {len(comparison['difference_details']) - 10} more differences")
                        
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '7':
            from answer_key_manager import create_sample_answer_keys
            create_sample_answer_keys()
        
        elif choice == '8':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode
        print("OMR Detection System - Interactive Mode")
        print("Choose an option:")
        print("1. Detect OMR from image")
        print("2. Run calibration tool")
        print("3. Manage answer keys")
        
        choice = input("\\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            answer_key = input("Enter answer key name (optional): ").strip() or None
            threshold = float(input("Enter detection threshold (0.65): ") or 0.65)
            debug = input("Enable debug mode? (y/N): ").strip().lower() == 'y'
            
            class Args:
                def __init__(self):
                    self.image = image_path
                    self.answer_key = answer_key
                    self.threshold = threshold
                    self.debug = debug
                    self.output = None
            
            detect_omr(Args())
        
        elif choice == '2':
            run_calibration()
        
        elif choice == '3':
            manage_answer_keys()
    
    else:
        # Command line mode
        main()