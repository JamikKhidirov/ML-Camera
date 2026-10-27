@echo off
chcp 65001 >nul
rem =====================================================================
rem  ПЕРВАЯ НАСТРОЙКА ПРОЕКТА НА ПК (запускается один раз).
rem  Ставит все нужные библиотеки и проверяет модели.
rem  Подробнее: docs/НАЧАЛО.md
rem =====================================================================
cd /d "%~dp0.."

echo [1/5] Создаю виртуальное окружение...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не найден Python. Поставь с https://python.org ^(галочка Add to PATH^).
        pause
        exit /b 1
    )
)

echo [2/5] Ставлю зависимости (долго, ~10 минут, один раз)...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-pc.txt

echo [3/5] Проверяю/скачиваю модели ИИ (YOLO, лица, марки авто)...
.venv\Scripts\python.exe scripts\download_models.py --config config\config.pc.yaml

echo [4/5] Ставлю нейросетевые номера (FastALPR: детектор + OCR)...
.venv\Scripts\python.exe scripts\install_alpr.py

echo [5/5] Создаю папки людей и запоминаю лица (ПК-конфиг)...
.venv\Scripts\python.exe main.py --config config\config.pc.yaml --register

echo.
echo ГОТОВО! Дальше:
echo   1) Открой  config\config.pc.yaml  и впиши токен бота, id, камеры  ^(см. docs\НАЧАЛО.md^)
echo   2) Положи фото в data\datasets\raw\ИМЯ\
echo   3) Запусти:  pc\run_pc.bat   (или python pc\main.py)
echo   4) Хочешь .exe:  pc\build_exe.bat
echo.
pause