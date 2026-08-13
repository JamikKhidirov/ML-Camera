@echo off
REM ============================================================
REM  MLCamera — запуск на ПК (Windows 11, Ryzen, RX 5700 XT)
REM  Открывает живое окно с видео и рамками, пишет видео в
REM  data\videos\<камера>\ и шлёт уведомления в Telegram/ntfy.
REM
REM  Подготовка (первый раз):
REM    1) python scripts\make_dirs.py
REM    2) python scripts\add_person.py Имя   (для каждого человека)
REM    3) положи фото в data\datasets\raw\<имя>\
REM    4) python scripts\collect_faces.py
REM    5) python scripts\register_faces.py
REM    6) открой config\config.pc.yaml и впиши: IP камер, токен бота,
REM       chat_ids (кому слать «кто пришёл»), ntfy topic
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Ошибка: python не найден в PATH. Установи Python 3.10+.
    pause
    exit /b 1
)

echo ============================================================
echo   MLCamera — умная камера (ПК-версия)
echo   Окно предпросмотра: включено
echo   Запись видео: data\videos\
echo   Чтобы выйти из окна: нажми Q
echo ============================================================

REM запуск с предпросмотром и записью
python main.py --preview

echo.
echo Программа завершилась. Если это ошибка — покажи текст выше.
pause