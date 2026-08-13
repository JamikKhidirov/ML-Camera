@echo off
REM ============================================================
REM  MLCamera — тесты + самопроверка всех модулей
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === 1/2 Автотесты (pytest) ===
python -m pytest tests -v

echo.
echo === 2/2 Самопроверка всех модулей ===
python scripts\self_check.py

pause