@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "CHECK_ONLY=0"
if /i "%~1"=="--check" set "CHECK_ONLY=1"
if not "%~1"=="" if /i not "%~1"=="--check" (
    echo ERROR: Only --check is supported.
    exit /b 2
)
set "APP_FILE=app\成本测算.py"
set "IMPORT_CHECK=import streamlit, pandas, numpy, joblib, lightgbm, sklearn, plotly; from sklearn.impute import SimpleImputer; import models.cost_model, models.coal_models, models.scenario_cost, app.scenario_view; assert tuple(map(int, streamlit.__version__.split('.')[:2])) >= (1, 50); assert hasattr(SimpleImputer, 'set_output')"
if not exist "%APP_FILE%" goto :missing_file
echo 火电机组成本测算研究原型
echo Selected: %APP_FILE%

if not exist ".venv\Scripts\python.exe" goto :find_conda
".venv\Scripts\python.exe" -c "%IMPORT_CHECK%" >nul 2>&1
if errorlevel 1 goto :find_conda
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
goto :run_python

:find_conda
set "CONDA_EXE="
for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%C"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE goto :find_python
set "CONDA_ENV="
for %%C in ("%CONDA_EXE%") do set "PATH=%%~dpC;%PATH%"
for /f "tokens=1" %%E in ('conda.exe env list 2^>nul ^| findstr /R /V "^#"') do call :try_conda "%%E"
if defined CONDA_ENV goto :run_conda

:find_python
where python >nul 2>&1
if errorlevel 1 goto :find_py
python -c "%IMPORT_CHECK%" >nul 2>&1
if errorlevel 1 goto :find_py
set "PYTHON_EXE=python"
goto :run_python

:find_py
where py >nul 2>&1
if errorlevel 1 goto :missing_environment
py -3 -c "%IMPORT_CHECK%" >nul 2>&1
if errorlevel 1 goto :missing_environment
if "%CHECK_ONLY%"=="1" goto :check_ok
py -3 -m streamlit run "%APP_FILE%" --server.address 127.0.0.1 --server.headless false
set "LAUNCH_STATUS=%errorlevel%"
goto :finished

:run_python
echo Python: %PYTHON_EXE%
if "%CHECK_ONLY%"=="1" goto :check_ok
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.address 127.0.0.1 --server.headless false
set "LAUNCH_STATUS=%errorlevel%"
goto :finished

:run_conda
echo Conda environment: %CONDA_ENV%
if "%CHECK_ONLY%"=="1" goto :check_ok
"%CONDA_EXE%" run --no-capture-output -n "%CONDA_ENV%" python -m streamlit run "%APP_FILE%" --server.address 127.0.0.1 --server.headless false
set "LAUNCH_STATUS=%errorlevel%"
goto :finished

:try_conda
if defined CONDA_ENV exit /b 0
"%CONDA_EXE%" run -n "%~1" python -c "%IMPORT_CHECK%" >nul 2>&1
if not errorlevel 1 set "CONDA_ENV=%~1"
exit /b 0

:check_ok
echo Environment check passed. Server was not started.
exit /b 0

:missing_file
echo ERROR: Application file not found: %APP_FILE%
goto :failed

:missing_environment
echo 未找到具备依赖的 Python 环境，需要 Streamlit 1.50 或以上。
echo 请在 Python 终端中进入本目录，按 README 安装 requirements.txt。
echo 本脚本不会自动安装软件。Conda 不在标准位置时，请从 Anaconda Prompt 运行。
goto :failed

:failed
if "%CHECK_ONLY%"=="0" pause
exit /b 1

:finished
if not "%LAUNCH_STATUS%"=="0" echo 启动或运行失败，请查看上方错误信息。
pause
exit /b %LAUNCH_STATUS%
