@echo off
set ROOT_DIR=%~dp0..
echo Starting Bugg AI Services...

:: Start the FastAPI Backend on Port 8001
echo Starting Bugg AI Backend...
cd /d "%ROOT_DIR%\backend"
start "Bugg AI Backend" cmd /k "python api.py"

:: Start the React Frontend
echo Starting Bugg AI Frontend...
cd /d "%ROOT_DIR%\frontend"
start "Bugg AI Frontend" cmd /k "npm run dev"

echo.
echo Both services are starting in separate windows.
echo Backend API: http://localhost:8001
echo React UI:   Check your browser for the URL (usually http://localhost:5173 for Vite)
echo.
echo If the UI shows "Neural link failed", wait a few seconds for the backend to initialize.
echo.
pause
