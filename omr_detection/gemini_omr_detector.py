import google.generativeai as genai
import PIL.Image
import sys
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyA2cqiH1MecukxgSyMtZ9K2zSZG3O3Rkoo")

def analyze_omr_image(image_path):
    """
    Analyze an OMR image using Gemini AI to detect marked answers.
    """
    try:
        # Load the image
        image = PIL.Image.open(image_path)
        
        # Create the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt for OMR analysis
        prompt = """
        Analyze this OMR (Optical Mark Recognition) sheet image. 
        Identify which bubbles/options are filled/marked for each question.
        Look for darkened circles or marked areas that indicate selected answers.
        Provide the results in a clear format like:
        Question 1: A
        Question 2: B
        Question 3: C
        etc.
        
        If a question has multiple marks, unclear marks, or no marks, please note it.
        Focus on the answer section of the OMR sheet.
        """
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        return response.text
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python gemini_omr_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing OMR image: {image_path}")
    print("=" * 50)
    
    result = analyze_omr_image(image_path)
    print(result)

if __name__ == "__main__":
    main()