@echo off
REM Set the script to stop on error
setlocal enabledelayedexpansion

echo Compiling Cellular Automaton Simulation...

REM Move to the parent directory (root of the project)

REM Define the root directory as the target for the executable
set ROOT_DIR=%cd%

REM Run PyInstaller with options
pyinstaller --onefile ^
    --distpath "%ROOT_DIR%" ^
    --add-data "GameOfLife.py;GameOfLife" ^
    --add-data "GeneticAlgorithm.py;GeneticAlgorithm.py" ^
    --add-data "InteractiveSimulation.py;InteractiveSimulation" ^
    --add-data "main.py" ^
    main.py

REM Notify the user
if %errorlevel% equ 0 (
    echo Compilation successful! The executable has been created in the root directory.
    echo File: "%ROOT_DIR%\main.exe"
) else (
    echo Compilation failed. Please check the logs for details.
)

REM Keep the terminal open
pause
