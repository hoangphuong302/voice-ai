@echo off
title Voice AI - F5-TTS Server
echo ============================================================
echo   Voice AI - Starting F5-TTS Local Server on port 9880
echo   Press Ctrl+C to stop
echo ============================================================
py -3.10 f5_api.py
pause
