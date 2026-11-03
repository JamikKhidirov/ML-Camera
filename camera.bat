@echo off
chcp 65001 >nul
cd /d "%~dp0"
title MLCamera — камера

rem Окно Tkinter: живой поток, рамки и скелет лица (106 точек).
rem FPS показа берётся из preview.fps в config\config.yaml (по умолчанию 30).
rem Закрыть: Esc или крестик. Снимок: S. Пауза: пробел. Полный экран: F11.

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "scripts\camera_view.py"
    exit /b 0
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "scripts\camera_view.py"
    if errorlevel 1 pause
    exit /b 0
)
python "scripts\camera_view.py"
if errorlevel 1 pause
