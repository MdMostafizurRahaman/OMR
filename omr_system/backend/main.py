from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from datetime import datetime
import pandas as pd
from typing import List, Optional
import json

from app.omr_processor import OMRProcessor
from app.answer_key_manager import AnswerKeyManager
from app.evaluator import Evaluator

app = FastAPI(title="OMR Processing System", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
omr_processor = OMRProcessor()
answer_key_manager = AnswerKeyManager()
evaluator = Evaluator()

# Create necessary directories
os.makedirs("uploads/omr_sheets", exist_ok=True)
os.makedirs("uploads/answer_keys", exist_ok=True)
os.makedirs("data", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "OMR Processing System API"}

@app.post("/upload-omr")
async def upload_omr(
    file: UploadFile = File(...),
    branch: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    exam_date: str = Form(...),
    set_code: Optional[str] = Form(None)
):
    """Upload OMR answer sheet for processing"""
    try:
        # Validate file type
        if not file.content_type in ["image/jpeg", "image/png", "application/pdf"]:
            raise HTTPException(status_code=400, detail="Only JPG, PNG, and PDF files are allowed")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        saved_filename = f"{file_id}.{file_extension}"
        file_path = f"uploads/omr_sheets/{saved_filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process OMR
        extracted_data = omr_processor.process_omr(file_path)
        
        # Add metadata
        extracted_data.update({
            "file_id": file_id,
            "original_filename": file.filename,
            "branch": branch,
            "class": class_name,
            "subject": subject,
            "exam_date": exam_date,
            "set_code": set_code,
            "upload_time": datetime.now().isoformat(),
            "file_path": file_path
        })
        
        # Save extracted data to CSV
        save_extracted_data(extracted_data)
        
        return JSONResponse({
            "success": True,
            "message": "OMR processed successfully",
            "data": extracted_data
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error processing OMR: {str(e)}"
        }, status_code=500)

@app.post("/upload-answer-key")
async def upload_answer_key(
    branch: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    set_code: str = Form(...),
    total_questions: int = Form(...),
    answers: str = Form(...)  # JSON string of answers
):
    """Upload answer key for a specific question set"""
    try:
        # Parse answers JSON
        answer_dict = json.loads(answers)
        
        # Validate answers
        if len(answer_dict) != total_questions:
            raise HTTPException(status_code=400, detail="Number of answers doesn't match total questions")
        
        # Save answer key
        answer_key_data = {
            "branch": branch,
            "class": class_name,
            "subject": subject,
            "set_code": set_code,
            "total_questions": total_questions,
            "answers": answer_dict,
            "created_time": datetime.now().isoformat()
        }
        
        answer_key_manager.save_answer_key(answer_key_data)
        
        return JSONResponse({
            "success": True,
            "message": "Answer key saved successfully",
            "data": answer_key_data
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error saving answer key: {str(e)}"
        }, status_code=500)

@app.post("/evaluate-omr")
async def evaluate_omr(
    file_id: str = Form(...),
    answer_key_id: str = Form(...)
):
    """Evaluate OMR against answer key"""
    try:
        # Get OMR data
        omr_data = get_omr_data(file_id)
        if not omr_data:
            raise HTTPException(status_code=404, detail="OMR data not found")
        
        # Get answer key
        answer_key = answer_key_manager.get_answer_key(answer_key_id)
        if not answer_key:
            raise HTTPException(status_code=404, detail="Answer key not found")
        
        # Evaluate
        result = evaluator.evaluate(omr_data, answer_key)
        
        # Save result
        save_evaluation_result(result)
        
        return JSONResponse({
            "success": True,
            "message": "Evaluation completed successfully",
            "result": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error during evaluation: {str(e)}"
        }, status_code=500)

@app.get("/answer-keys")
async def get_answer_keys():
    """Get list of all answer keys"""
    try:
        answer_keys = answer_key_manager.get_all_answer_keys()
        return JSONResponse({
            "success": True,
            "data": answer_keys
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error fetching answer keys: {str(e)}"
        }, status_code=500)

@app.get("/omr-data")
async def get_all_omr_data():
    """Get list of all processed OMR data"""
    try:
        # Read from CSV file
        csv_path = "data/omr_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return JSONResponse({
                "success": True,
                "data": df.to_dict('records')
            })
        else:
            return JSONResponse({
                "success": True,
                "data": []
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error fetching OMR data: {str(e)}"
        }, status_code=500)

@app.get("/results")
async def get_results():
    """Get evaluation results"""
    try:
        results_path = "data/evaluation_results.csv"
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            return JSONResponse({
                "success": True,
                "data": df.to_dict('records')
            })
        else:
            return JSONResponse({
                "success": True,
                "data": []
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error fetching results: {str(e)}"
        }, status_code=500)

def save_extracted_data(data):
    """Save extracted OMR data to CSV"""
    csv_path = "data/omr_data.csv"
    
    # Convert to DataFrame
    df_new = pd.DataFrame([data])
    
    # Append or create CSV
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)

def get_omr_data(file_id):
    """Get OMR data by file ID"""
    csv_path = "data/omr_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        matching_rows = df[df['file_id'] == file_id]
        if not matching_rows.empty:
            return matching_rows.iloc[0].to_dict()
    return None

def save_evaluation_result(result):
    """Save evaluation result to CSV"""
    csv_path = "data/evaluation_results.csv"
    
    # Convert to DataFrame
    df_new = pd.DataFrame([result])
    
    # Append or create CSV
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)