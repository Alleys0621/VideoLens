@echo off
chcp 65001 >nul
REM ============================================================
REM  VideoLens One-Click Start Script
REM  - Double-click to run, or run from cmd: start.bat
REM  - Stop: close the minimized service windows from taskbar AND this main window
REM
REM  Startup order:
REM    [0/6] Cleanup  : kill stale processes on ports 3000/2024/9800/9801
REM    [1/6] Postgres : docker compose up (port 5432)
REM    [2/6] Backend  : LangGraph (port 2024)
REM    [3/6] Frontend : Next.js (port 3000)
REM    [4/6] ASR      : paraformer streaming (port 9800)
REM    [5/6] TTS      : qwen-audio-tts streaming (port 9801)
REM    [6/6] Tunnel   : cloudflared quick tunnel (random *.trycloudflare.com)
REM ============================================================

set "PYTHONUTF8=1"
set "LANGSMITH_TRACING=false"
set "LANGCHAIN_TRACING=false"

REM Postgres connection (与 db/docker-compose.yml 一致; LangGraph checkpointer + 前端 DAO 共用)
set "POSTGRES_URL=postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens"

REM TLS 证书 (mkcert 签的本地证书; 覆盖 localhost / 127.0.0.1 / Alleys / Alleys.local).
REM 各服务 (ASR/TTS/视频) 通过 VIDEOLENS_TLS_CERT/KEY env 启用 wss/https.
set "VIDEOLENS_TLS_CERT=%~dp0certs\localhost+4.pem"
set "VIDEOLENS_TLS_KEY=%~dp0certs\localhost+4-key.pem"

cd /d %~dp0

echo ============================================================
echo   VideoLens Starting
echo   Postgres : localhost:25432              (docker compose)
echo   Backend  : http://localhost:2024        (LangGraph)
echo   Frontend : http://0.0.0.0:3000          (Next.js, 局域网可访问)
echo   ASR      : ws://0.0.0.0:9800            (paraformer)
echo   TTS      : ws://0.0.0.0:9801            (qwen-audio-tts)
echo   Video    : 通过 Next.js /api/video 路由 (不再依赖 9802)
echo   Tunnel   : cloudflared (random URL per launch)
echo ============================================================
echo.

REM === [0/6] Cleanup: kill stale processes ===
echo [0/6] Cleaning up stale processes...

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

REM === [1/6] Postgres (docker compose) ===
echo [1/6] Starting Postgres (docker compose)...
docker compose -f db/docker-compose.yml up -d
if errorlevel 1 (
    echo.
    echo [ERROR] docker compose 失败. 请确认 Docker Desktop 已启动.
    echo [ERROR] 启动 Docker Desktop 后重试.
    echo.
    pause
    exit /b 1
)
REM 等 Postgres 就绪 (最长 ~30s)
echo       Waiting for Postgres to accept connections...
set "PG_READY=0"
for /l %%i in (1,1,15) do (
    docker exec videolens-postgres pg_isready -U videolens -d videolens >nul 2>&1
    if not errorlevel 1 (
        set "PG_READY=1"
        goto :PG_OK
    )
    ping 127.0.0.1 -n 3 >nul
)
:PG_OK
if "%PG_READY%"=="0" (
    echo [WARN] Postgres 未就绪, 后续步骤可能失败
) else (
    echo       Postgres ready.
)

REM [2/6] Backend - LangGraph Server
REM /min: 最小化窗口启动, 避免一次性弹出多个终端
echo [2/6] Starting Backend LangGraph (port 2024, minimized)...
start /min "VideoLens-Backend-2024" cmd /k "set POSTGRES_URL=%POSTGRES_URL% && .venv\Scripts\langgraph.exe dev --port 2024 --no-browser"

REM [3/6] Frontend - Next.js (HTTPS via mkcert)
REM -H 0.0.0.0: 同局域网设备通过 https://Alleys.local:3000 访问 (协作者需先装 mkcert CA)
REM --experimental-https + cert/key: 启用 HTTPS, 满足浏览器 SecureContext 要求 (getUserMedia/AudioWorklet)
echo [3/6] Starting Frontend Next.js (HTTPS port 3000, host 0.0.0.0, minimized)...
start /min "VideoLens-Frontend-3000" cmd /k "cd frontend && node_modules\.bin\next.CMD dev -H 0.0.0.0 --experimental-https --experimental-https-key ..\certs\localhost+4-key.pem --experimental-https-cert ..\certs\localhost+4.pem"

REM [4/6] ASR WebSocket server (streaming speech recognition)
echo [4/6] Starting ASR server (port 9800, minimized)...
start /min "VideoLens-ASR-Server" cmd /k ".venv\Scripts\python.exe -m src.agent.asr_server"

REM [5/6] TTS WebSocket server (streaming speech synthesis)
echo [5/6] Starting TTS server (port 9801, minimized)...
start /min "VideoLens-TTS-Server" cmd /k ".venv\Scripts\python.exe -m src.agent.tts_server"

REM [5.5/6] Video static server (独立于 Next.js, 避免 next dev JIT 影响视频流)
echo [5.5/6] Starting video static server (port 9802, minimized)...
start /min "VideoLens-Video-Server" cmd /k ".venv\Scripts\python.exe scripts\_video_server.py"

REM [6/6] cloudflared quick tunnel
set "CF_BIN="
if exist ".tools\cloudflared.exe" (
    set "CF_BIN=.tools\cloudflared.exe"
) else if exist ".tools\cloudflared-windows-amd64.exe" (
    set "CF_BIN=.tools\cloudflared-windows-amd64.exe"
)
if not defined CF_BIN (
    echo.
    echo [WARN] cloudflared not found in .tools/. Skipping tunnel.
    echo [WARN] Run scripts\install-cloudflared.bat, or put cloudflared.exe there.
    echo [WARN] On same WiFi use http://YOUR_IP:3000
    echo.
    goto :WAIT_END
)

echo [6/6] Starting cloudflared quick tunnel (minimized)...
REM cloudflared quick tunnel: random URL per launch, printed in the minimized window.
REM To stop: close the "VideoLens-Tunnel" window from taskbar.
start /min "VideoLens-Tunnel" cmd /k "%CF_BIN% tunnel --url http://127.0.0.1:3000"

:WAIT_END
echo.
echo ============================================================
echo   All services started (minimized to taskbar):
echo     - Postgres : container "videolens-postgres" (docker ps)
echo     - Backend  : minimized "VideoLens-Backend-2024"
echo     - Frontend : minimized "VideoLens-Frontend-3000"
echo     - ASR      : minimized "VideoLens-ASR-Server"
echo     - TTS      : minimized "VideoLens-TTS-Server"
echo     - Tunnel   : minimized "VideoLens-Tunnel" (URL printed there)
echo.
echo   Access:
echo     [Local]    http://localhost:3000
echo     [LAN]      http://本机IP:3000  (同 WiFi 设备直接访问; ASR/TTS 走 IP:9800/9801)
echo     [Public]   See the URL in "VideoLens-Tunnel" window
echo                (changes each launch, e.g. https://xxx.trycloudflare.com)
echo                NOTE: 公网下 ASR/TTS 语音不可用 (cloudflared 单隧道未反代 ws);
echo                      页面/文字对话/视频可正常使用
echo.
echo   Stop: close minimized windows from taskbar AND this main window
echo ============================================================
echo.
pause
