@echo off
chcp 65001
echo 启动中，请耐心等待AIDOBE@https://space.bilibili.com/554863038/@passionlo

SET PYTHON_HOME=%cd%\myvenv
SET PYTHON_PATH=%PYTHON_HOME%
SET SC_PATH=%PYTHON_HOME%\Scripts
SET PATH=%PYTHON_HOME%;%PYTHON_HOME%\Lib\site-packages;%SC_PATH%;%PATH%
"%PYTHON_HOME%\python.exe" app.py
pause
