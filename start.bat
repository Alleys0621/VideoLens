@echo off
chcp 65001 >nul
REM ============================================================
REM  VideoLens One-Click Start Script
REM  - Double-click to run, or run from cmd: start.bat
REM  - Stop: close the popup cmd windows
REM
REM  Startup order:
REM    [0/5] Cleanup  : kill stale processes on ports 3000/2024/9800/9801
REM    [1/5] Backend  : LangGraph (port 2024)
REM    [2/5] Frontend : Next.js (port 3000)
REM    [3/5] ASR      : paraformer streaming (port 8000)
REM    [4/5] TTS      : qwen-audio-tts streaming (port 8001)
REM    [5/5] Tunnel   : cloudflared quick tunnel (random *.trycloudflare.com)
REM ============================================================

set "PYTHONUTF8=1"
set "LANGSMITH_TRACING=false"
set "LANGCHAIN_TRACING=false"

cd /d %~dp0

echo ============================================================
echo   VideoLens Starting
echo   Backend  : http://localhost:2024  (LangGraph)
echo   Frontend : http://localhost:3000  (Next.js)
echo   ASR      : ws://localhost:9800    (paraformer)
echo   TTS      : ws://localhost:9801    (qwen-audio-tts)
echo   Tunnel   : cloudflared (random URL per launch)
echo ============================================================
echo.

REM === [0/5] Cleanup: kill stale processes ===
echo [0/5] Cleaning up stale processes...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000 " ^| findstr LISTENING') do (
    echo       Killing PID %%a (port 3000)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":2024 " ^| findstr LISTENING') do (
    echo       Killing PID %%a (port 2024)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9800 " ^| findstr LISTENING') do (
    echo       Killing PID %%a (port 9800)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9801 " ^| findstr LISTENING') do (
    echo       Killing PID %%a (port 9801)
    taskkill /F /PID %%a >nul 2>&1
)

tasklist | findstr /i "cloudflared.exe" >nul 2>&1
if not errorlevel 1 (
    echo       Killing stale cloudflared.exe
    taskkill /F /IM cloudflared.exe >nul 2>&1
)

ping 127.0.0.1 -n 3 >nul

REM [0.5/5] 初始化用户数据库 (从模板复制, 如果不存在)
if not exist ".langgraph_api\store.pckl" (
    if exist ".langgraph_api\store.pckl.template" (
        echo [init] Copying store.pckl from template...
        copy ".langgraph_api\store.pckl.template" ".langgraph_api\store.pckl" >nul
    )
)

REM [1/5] Backend - LangGraph Server
echo [1/5] Starting Backend LangGraph (port 2024)...
start "VideoLens-Backend-2024" cmd /k ".venv\Scripts\langgraph.exe dev --port 2024 --no-browser"

REM [2/5] Frontend - Next.js
echo [2/5] Starting Frontend Next.js (port 3000)...
start "VideoLens-Frontend-3000" cmd /k "cd frontend && node_modules\.bin\next.CMD dev"

REM [3/5] ASR WebSocket server (streaming speech recognition)
echo [3/5] Starting ASR server (port 9800, paraformer streaming)...
start "VideoLens-ASR-Server" cmd /k ".venv\Scripts\python.exe -m src.agent.asr_server"

REM [4/5] TTS WebSocket server (streaming speech synthesis)
echo [4/5] Starting TTS server (port 9801, qwen-audio-tts streaming)...
start "VideoLens-TTS-Server" cmd /k ".venv\Scripts\python.exe -m src.agent.tts_server"

REM [5/5] cloudflared quick tunnel
if not exist ".tools\cloudflared.exe" (
    echo.
    echo [WARN] .tools\cloudflared.exe not found. Skipping tunnel.
    echo [WARN] On same WiFi use http://YOUR_IP:3000
    echo.
    goto :WAIT_END
)

echo [5/5] Starting cloudflared quick tunnel...
REM cloudflared quick tunnel: random URL per launch, printed in the popup window.
REM To stop: close the "VideoLens-Tunnel" popup window.
start "VideoLens-Tunnel" cmd /k ".tools\cloudflared.exe tunnel --url http://127.0.0.1:3000"

:WAIT_END
echo.
echo ============================================================
echo   All services started:
echo     - Backend  : popup "VideoLens-Backend-2024"
echo     - Frontend : popup "VideoLens-Frontend-3000"
echo     - ASR      : popup "VideoLens-ASR-Server"
echo     - TTS      : popup "VideoLens-TTS-Server"
echo     - Tunnel   : popup "VideoLens-Tunnel" (URL printed there)
echo.
echo   Access:
echo     [Local]    http://localhost:3000
echo     [Public]   See the URL in "VideoLens-Tunnel" window
echo                (changes each launch, e.g. https://xxx.trycloudflare.com)
echo.
echo   Stop: close popup windows AND this main window
echo ============================================================
echo.
pause
