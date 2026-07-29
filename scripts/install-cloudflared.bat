@echo off
chcp 65001 >nul
REM 一键下载 cloudflared.exe（Windows amd64）到 .tools/
REM 不需要公网隧道可跳过；start.bat / start-prod.bat 会自动检测并跳过

cd /d %~dp0\..

if exist ".tools\cloudflared.exe" (
    echo [INFO] .tools\cloudflared.exe 已存在，跳过下载。
    goto :VERIFY
)

echo [INFO] 正在下载最新版 cloudflared-windows-amd64.exe ...
if not exist ".tools" mkdir ".tools"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '.tools\cloudflared.exe' -UseBasicParsing } catch { Write-Error $_; exit 1 }"

if errorlevel 1 (
    echo [ERROR] 下载失败。请手动从 https://github.com/cloudflare/cloudflared/releases 下载 cloudflared-windows-amd64.exe，
    echo         重命名为 cloudflared.exe 放到 .tools\ 目录下。
    exit /b 1
)

:VERIFY
".tools\cloudflared.exe" --version
if errorlevel 1 (
    echo [ERROR] .tools\cloudflared.exe 校验失败，请重新下载。
    exit /b 1
)

echo [OK] cloudflared 已就绪：.tools\cloudflared.exe
pause
