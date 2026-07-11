@echo off
REM ============================================================
REM  VideoLens One-Click Start Script
REM  - Double-click to run, or run from cmd: start.bat
REM  - Edit this file to change ports/commands
REM  - Stop: close the two popup cmd windows
REM ============================================================

REM PYTHONUTF8=1 (quoted to avoid trailing space — Python rejects "1 ")
REM Set globally here; child cmd windows inherit it. Harmless for node/frontend.
set "PYTHONUTF8=1"

REM 关闭 LangSmith tracing (当前 MVP 不需要, 省网络开销 + 启动不连 LangSmith)
set "LANGSMITH_TRACING=false"
set "LANGCHAIN_TRACING=false"

cd /d %~dp0

echo ============================================================
echo   VideoLens Starting...
echo   Backend  : http://localhost:2024  (LangGraph, runs src/server/graph.py)
echo   Frontend : http://localhost:3000  (Next.js)
echo ============================================================
echo.

REM [1/2] Backend - LangGraph Server
REM   Full path to .venv\langgraph.exe avoids PATH not found (exit 127)
REM   No need to "activate" venv: .exe has absolute Python baked in
echo [1/2] Starting Backend LangGraph (port 2024)...
start "VideoLens-Backend-2024" cmd /k ".venv\Scripts\langgraph.exe dev --port 2024"

REM [2/2] Frontend - Next.js
echo [2/2] Starting Frontend Next.js (port 3000)...
start "VideoLens-Frontend-3000" cmd /k "cd frontend && npx pnpm dev"

echo.
echo ============================================================
echo   Both services started in separate windows.
echo   Open http://localhost:3000 when both windows show Ready.
echo   Stop: close the popup windows.
echo ============================================================
echo.
pause
