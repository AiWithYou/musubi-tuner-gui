@echo off
chcp 65001 > nul
cd /d %~dp0
cd ..\musubi-tuner

echo Starting GUI... This may take a while for the first time.
echo GUIを起動しています... 
uv run --extra cu124 --extra gui python ..\gui\gui.py

pause
