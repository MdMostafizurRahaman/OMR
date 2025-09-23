# OMR Processing System

An automated OMR (Optical Mark Recognition) processing system for coaching centers with multiple branches. The system allows teachers to upload answer keys and students' OMR answer sheets, automatically processes them, and generates detailed evaluation results.

## Features

- **Multi-branch Support**: Different branches, classes, and subjects
- **Automatic OMR Processing**: Extract roll numbers, set codes, and answers from scanned OMR sheets
- **Manual Entry Fallback**: If automatic extraction fails, manual answer entry is available
- **Answer Key Management**: Upload and manage answer keys for different question sets
- **Detailed Evaluation**: Compare student answers with answer keys and generate comprehensive results
- **Performance Analytics**: View detailed statistics including correct, wrong, and blank answers
- **Export Data**: Save extracted data and results in CSV format

## Technology Stack

- **Frontend**: React.js with Bootstrap for responsive UI
- **Backend**: FastAPI (Python) for REST API
- **Image Processing**: OpenCV and Tesseract OCR for OMR processing
- **Data Storage**: CSV files for data persistence (can be upgraded to database)

## Project Structure

```
omr_system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── omr_processor.py      # OMR image processing logic
│   │   ├── answer_key_manager.py # Answer key management
│   │   └── evaluator.py          # Evaluation logic
│   ├── uploads/
│   │   ├── omr_sheets/           # Uploaded OMR images
│   │   └── answer_keys/          # Answer key files
│   ├── data/                     # CSV data files
│   ├── main.py                   # FastAPI application
│   └── requirements.txt          # Python dependencies
└── frontend/
    ├── src/
    │   ├── components/           # React components
    │   ├── services/             # API services
    │   └── pages/                # Page components
    ├── package.json              # Node.js dependencies
    └── public/                   # Static files
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js 14+
- Tesseract OCR (for Windows: https://github.com/UB-Mannheim/tesseract/wiki)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

5. Run the backend:
```bash
python main.py
```

Backend will be available at: http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

Frontend will be available at: http://localhost:3000

## Usage

### 1. Upload Answer Key

1. Go to "Answer Keys" section
2. Fill in the question set information (branch, class, subject, set code)
3. Enter the correct answers for all questions
4. Save the answer key

### 2. Upload OMR Answer Sheets

1. Go to "Upload OMR" section
2. Fill in the exam information
3. Upload the OMR image or PDF
4. The system will automatically extract:
   - Roll number
   - Set code
   - Student answers
5. If automatic extraction fails, you can enter answers manually

### 3. Evaluate Results

1. Go to "Results" section
2. Select the OMR data and corresponding answer key
3. Click "Evaluate" to generate results
4. View detailed performance analysis

## API Endpoints

- `POST /upload-omr` - Upload OMR answer sheet
- `POST /upload-answer-key` - Upload answer key
- `POST /evaluate-omr` - Evaluate OMR against answer key
- `GET /answer-keys` - Get all answer keys
- `GET /omr-data` - Get all OMR data
- `GET /results` - Get evaluation results

## Data Format

### OMR Data CSV
- file_id, roll_number, branch, class, subject, exam_date, set_code, answers, total_answers, processing_method

### Answer Key JSON
- branch, class, subject, set_code, total_questions, answers, created_time

### Evaluation Results CSV
- evaluation_id, roll_number, branch, class, subject, correct_answers, wrong_answers, blank_answers, total_marks, percentage

## Customization

### Marking Scheme
Edit the `evaluator.py` file to customize:
- Marks per question
- Negative marking
- Grade boundaries

### OMR Processing
Edit the `omr_processor.py` file to:
- Adjust image preprocessing parameters
- Modify bubble detection algorithms
- Add new extraction patterns

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Make sure Tesseract is installed and added to PATH
2. **Image processing errors**: Check image quality and format (JPG, PNG, PDF supported)
3. **CORS errors**: Backend must be running on port 8000, frontend on port 3000

### Tips for Better OMR Recognition

1. Use high-quality scanned images (300 DPI or higher)
2. Ensure proper lighting and contrast
3. Avoid skewed or rotated images
4. Use clear, dark marks in bubbles

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support or questions, please contact the development team or create an issue in the repository.