@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title MLCamera — настройки

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "scripts\settings.py"
    exit /b 0
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "scripts\settings.py"
    if errorlevel 1 pause
    exit /b 0
)
python "scripts\settings.py"
if errorlevel 1 pause
