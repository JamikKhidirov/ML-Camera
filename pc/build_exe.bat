@echo off
chcp 65001 >nul
rem =====================================================================
rem  СБОРКА .exe для ПК — делается ОДИН раз на своём компьютере.
rem  Результат: pc\dist\MLCamera\MLCamera.exe (двойной клик — и камера).
rem  Подробнее: docs/НАЧАЛО.md
rem =====================================================================
cd /d "%~dp0.."

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ОШИБКА] Сначала запусти pc\setup.bat  ^(создаёт окружение^)
    pause
    exit /b 1
)

echo [1/5] Обновляю pip и зависимости (первые минуты)...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements-build.txt

echo [2/5] Генерирую иконку...
"%PY%" scripts\make_icon.py

echo [3/5] Собираю MLCamera.exe (PyInstaller, ~5-15 минут)...
"%PY%" -m PyInstaller --noconfirm --clean --onedir --name MLCamera ^
    --icon pc\MLCamera.ico ^
    --distpath pc\dist --workpath build\pyi ^
    --collect-data insightface --collect-data ultralytics ^
    pc\main.py
if errorlevel 1 (
    echo [ОШИБКА] Сборка не удалась. См. выше.
    pause
    exit /b 1
)

echo [4/5] Кладу config\ и data\ рядом с exe (чтобы можно было дополнять)...
if not exist "pc\dist\MLCamera\config" xcopy /e /i /q /y config "pc\dist\MLCamera\config" >nul
if not exist "pc\dist\MLCamera\data"    xcopy /e /i /q /y data    "pc\dist\MLCamera\data" >nul

echo       + объекты InsightFace в _internal\objects\
if not exist "pc\dist\MLCamera\_internal\objects" mkdir "pc\dist\MLCamera\_internal\objects"
xcopy /q /y /i ".venv\Lib\site-packages\insightface\data\objects" "pc\dist\MLCamera\_internal\objects" >nul

echo [5/5] ГОТОВО!
echo.
echo    ПРОГРАММА:  pc\dist\MLCamera\MLCamera.exe
echo    Фотки людей:  pc\dist\MLCamera\data\datasets\raw\ИМЯ\
echo                 ^(положи туда 10-25 фото, потом запусти  MLCamera.exe --register^)
echo    Настройки:    pc\dist\MLCamera\config\config.yaml
echo.
pause