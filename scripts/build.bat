@echo off
setlocal
echo ==========================================================
echo   Building CPU Matrix Multiplication Accelerator (Windows)
echo ==========================================================

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set BUILD_DIR=%PROJECT_ROOT%\build_win

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake -DCMAKE_BUILD_TYPE=Release "%PROJECT_ROOT%"
cmake --build . --config Release

echo Build complete! Binaries located in %BUILD_DIR%\bin
endlocal
