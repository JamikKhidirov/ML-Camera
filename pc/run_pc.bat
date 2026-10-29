@echo off
chcp 65001 >nul
rem =====================================================================
rem  БЫСТРЫЙ ЗАПУСК КАМЕРЫ С КОМПЬЮТЕРА (окно с видео + запись + уведомления)
rem  = python main.py (ПК-версия config/config.yaml)
rem =====================================================================
cd /d "%~dp0.."

if not exist .venv\Scripts\python.exe (
    echo [ОШИБКА] Нет окружения. Сначала дважды кликни pc\setup.bat
    pause
    exit /b 1
)

.venv\Scripts\python.exe pc\main.py
pause