@echo off
echo Starting OMR Processing System...

echo.
echo Starting Backend (FastAPI)...
cd backend
start cmd /k "python main.py"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend (React)...
cd ..\frontend
start cmd /k "npm start"

echo.
echo System is starting up...
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:3000
echo.
pause