@echo off
chcp 65001 >nul
REM ============================================================
REM  VideoLens Production Start (no hot-reload, max throughput)
REM
REM  与 start.bat 的区别:
REM    - Backend: --no-reload (关闭 watchfiles, 消除热更新轮询开销)
REM    - 走 scripts/_langgraph_prod.py 入口 (patch uvicorn loop factory,
REM      绕过 Windows ProactorEventLoop + psycopg 兼容问题)
REM    - Frontend: next start (生产构建, 不是 next dev)
REM      首次运行前必须 cd frontend && pnpm build
REM
REM  适合: 给用户演示 / 性能测试 / 长时间稳定运行
REM  不适合: 开发迭代 (改代码不会自动 reload, 需手动重启)
REM ============================================================

set "PYTHONUTF8=1"
set "LANGSMITH_TRACING=false"
set "LANGCHAIN_TRACING=false"
set "POSTGRES_URL=postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens"

cd /d %~dp0

echo ============================================================
echo   VideoLens Production Start
echo   Postgres : localhost:25432              (docker compose)
echo   Backend  : http://localhost:2024        (LangGraph, no reload)
echo   Frontend : http://localhost:3000        (Next.js, production build)
echo   ASR      : ws://localhost:9800          (paraformer)
echo   TTS      : ws://localhost:9801          (qwen-audio-tts)
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
    echo [ERROR] docker compose failed. Check Docker Desktop is running.
    pause
    exit /b 1
)
echo       Waiting for Postgres...
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
    echo [WARN] Postgres not ready
) else (
    echo       Postgres ready.
)

REM === [2/6] Backend - LangGraph (no reload, via wrapper) ===
echo [2/6] Starting Backend LangGraph (port 2024, no reload)...
start /min "VideoLens-Prod-Backend-2024" cmd /k "set POSTGRES_URL=%POSTGRES_URL% && .venv\Scripts\python.exe -m scripts._langgraph_prod dev --port 2024 --no-browser --no-reload"

REM === [3/6] Frontend - Next.js production ===
if not exist "frontend\.next\BUILD_ID" (
    echo.
    echo [WARN] frontend\.next\BUILD_ID not found.
    echo [WARN] First run requires: cd frontend ^&^& pnpm build
    echo [WARN] Falling back to next dev for this launch.
    echo.
    start /min "VideoLens-Prod-Frontend-3000" cmd /k "cd frontend && node_modules\.bin\next.CMD dev"
) else (
    echo [3/6] Starting Frontend Next.js (port 3000, production build)...
    start /min "VideoLens-Prod-Frontend-3000" cmd /k "cd frontend && node_modules\.bin\next.CMD start"
)

REM === [4/6] ASR WebSocket server ===
echo [4/6] Starting ASR server (port 9800)...
start /min "VideoLens-Prod-ASR" cmd /k ".venv\Scripts\python.exe -m src.agent.asr_server"

REM === [5/6] TTS WebSocket server ===
echo [5/6] Starting TTS server (port 9801)...
start /min "VideoLens-Prod-TTS" cmd /k ".venv\Scripts\python.exe -m src.agent.tts_server"

REM === [6/6] cloudflared quick tunnel ===
if not exist ".tools\cloudflared.exe" (
    echo.
    echo [WARN] .tools\cloudflared.exe not found. Skipping tunnel.
    goto :WAIT_END
)

echo [6/6] Starting cloudflared quick tunnel...
start /min "VideoLens-Prod-Tunnel" cmd /k ".tools\cloudflared.exe tunnel --url http://127.0.0.1:3000"

:WAIT_END
echo.
echo ============================================================
echo   Production services started (minimized to taskbar):
echo     - Backend  : minimized "VideoLens-Prod-Backend-2024"
echo     - Frontend : minimized "VideoLens-Prod-Frontend-3000"
echo     - ASR      : minimized "VideoLens-Prod-ASR"
echo     - TTS      : minimized "VideoLens-Prod-TTS"
echo     - Tunnel   : minimized "VideoLens-Prod-Tunnel"
echo.
echo   Access:
echo     [Local]    http://localhost:3000
echo     [Public]   See URL in "VideoLens-Prod-Tunnel" window
echo.
echo   Stop: close minimized windows from taskbar AND this main window
echo ============================================================
echo.
pause
