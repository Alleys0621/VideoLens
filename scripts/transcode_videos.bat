@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ============================================================
REM VideoLens Batch Video Transcoder
REM
REM Converts ALL videos under data\videos\ to H.264+AAC MP4
REM Supported input: mkv/mp4/avi/mov/flv/ts/wmv/webm/m4v/
REM   rmvb/rm/mpeg/mpg/m2ts/vob/3gp/f4v
REM
REM Why: H.264+AAC in MP4 is the only combo that works on
REM   ALL browsers (Chrome/Safari/Firefox/Edge).
REM   MKV container, HEVC(H.265) video, MP2/AC3 audio all
REM   have browser compatibility issues.
REM
REM Output: same filename .mp4 in same directory
REM Original: kept as .bak (delete after verification)
REM
REM Requirements: ffmpeg in system PATH
REM Usage: double-click or run from command line
REM ============================================================

cd /d "%~dp0\.."

echo ============================================================
echo   VideoLens Video Transcoder
echo   Target: H.264 + AAC in MP4 (all-browser compatible)
echo   Scanning: data\videos\
echo ============================================================
echo.

REM Check ffmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ffmpeg not found in PATH!
  echo   Download: https://www.gyan.dev/ffmpeg/builds/
  echo   Add the bin folder to system PATH.
  pause
  exit /b 1
)

set COUNT=0
set SUCCESSED=0

for /r "data\videos" %%F in (*.mkv *.mp4 *.avi *.mov *.flv *.ts *.wmv *.webm *.m4v *.rmvb *.rm *.mpeg *.mpg *.m2ts *.vob *.3gp *.f4v) do (
  REM Skip .bak files
  echo "%%~nxF" | findstr /i "\.bak$" >nul
  if !errorlevel! equ 0 (
    echo   [SKIP] %%~nxF ^(backup^)
  ) else (
    set /a COUNT+=1
    echo [!COUNT!] Transcoding: %%~nxF

    ffmpeg -y -i "%%~fF" ^
      -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p ^
      -c:a aac -b:a 128k ^
      -movflags +faststart ^
      "%%~dpnF_converting.mp4" ^
      -loglevel warning 2>&1

    if exist "%%~dpnF_converting.mp4" (
      if exist "%%~fF.bak" del /q "%%~fF.bak"
      ren "%%~fF" "%%~nxF.bak"
      ren "%%~dpnF_converting.mp4" "%%~nF.mp4"
      echo     [OK] -^> %%~nF.mp4
      set /a SUCCESSED+=1
    ) else (
      echo     [FAIL] ffmpeg error
    )
    echo.
  )
)

echo ============================================================
echo   Done!
echo   Scanned: !COUNT! files
echo   Success: !SUCCESSED! files
echo.
echo   Backups saved as .bak
echo   Delete after verification:
echo     del /s /q "data\videos\*.bak"
echo ============================================================
pause
