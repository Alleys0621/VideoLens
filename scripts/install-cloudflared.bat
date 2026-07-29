@echo off
chcp 65001 >nul
REM 下载 cloudflared.exe 到 .tools/（可选，公网隧道用）

cd /d %~dp0\..

if exist ".tools\cloudflared.exe" (
    echo [INFO] .tools\cloudflared.exe 已存在，跳过下载。
    goto :VERIFY
)

echo [INFO] 正在下载 cloudflared-windows-amd64.exe ...
if not exist ".tools" mkdir ".tools"

curl.exe -L -o ".tools\cloudflared.exe" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
if errorlevel 1 (
    echo [ERROR] curl 下载失败，请手动从 https://github.com/cloudflare/cloudflared/releases 下载 cloudflared-windows-amd64.exe，重命名为 cloudflared.exe 放到 .tools\ 目录下。
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
