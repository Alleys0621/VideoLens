@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ============================================================
REM VideoLens 批量视频转码脚本
REM
REM 功能: 将 data/videos/ 下所有视频统一转成 H.264+AAC MP4
REM       覆盖格式: mkv/mp4/avi/mov/flv/ts/wmv/webm/m4v/rmvb/
rm/mpeg/mpg/m2ts/vob/3gp/f4v
REM
REM 解决问题:
REM   - MKV 容器 Safari/Firefox 不支持
REM   - HEVC(H.265) Chrome/Firefox 支持有限, 音画不同步
REM   - MP2/AC3/DTS 音频编码浏览器不支持
REM   - 10bit 色深兼容性差
REM
REM 转码参数:
REM   -c:v libx264 -crf 23 -preset fast   视频: H.264 高质量
REM   -pix_fmt yuv420p                      色彩: 8bit 兼容
REM   -c:a aac -b:a 128k                    音频: AAC 128kbps
REM   -movflags +faststart                  moov 前置, 支持边下边播
REM
REM 环境要求:
REM   - Windows
REM   - ffmpeg (系统 PATH 中可访问, 验证: ffmpeg -version)
REM   - 不需要 Python 环境
REM
REM 用法:
REM   方式 1: 双击 scripts\transcode_videos.bat
REM   方式 2: 命令行执行:  scripts\transcode_videos.bat
REM
REM 转码后:
REM   - 新文件: 同名 .mp4 (如 第01集.mp4)
REM   - 原文件: 保留为 .bak (如 第01集.mkv.bak)
REM   - 确认无误后删除备份: del /s /q "data\videos\*.bak"
REM ============================================================

cd /d "%~dp0\.."

echo ============================================================
echo   VideoLens 视频转码工具
echo   目标格式: H.264 + AAC in MP4 (全浏览器兼容)
echo   扫描目录: data\videos\
echo ============================================================
echo.

REM 检查 ffmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 ffmpeg! 请先安装 ffmpeg 并加入系统 PATH.
  echo   下载: https://www.gyan.dev/ffmpeg/builds/
  echo   安装: 解压后把 bin 目录加入系统环境变量 PATH
  pause
  exit /b 1
)

set COUNT=0
set SUCCESSED=0

for /r "data\videos" %%F in (*.mkv *.mp4 *.avi *.mov *.flv *.ts *.wmv *.webm *.m4v *.rmvb *.rm *.mpeg *.mpg *.m2ts *.vob *.3gp *.f4v) do (
  set /a COUNT+=1

  REM 跳过已经是转码产物 (.bak 文件)
  echo "%%~nxF" | findstr /i "\.bak$" >nul && (
    echo   [跳过] %%~nxF (备份文件)
    echo.
    goto :continue
  )

  echo [!COUNT!] 转码: %%~nxF
  echo     路径: %%~dpnxF

  REM 转码到临时文件
  ffmpeg -y -i "%%~fF" ^
    -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p ^
    -c:a aac -b:a 128k ^
    -movflags +faststart ^
    "%%~dpnF_converting.mp4" ^
    -loglevel warning 2>&1

  if exist "%%~dpnF_converting.mp4" (
    REM 原文件改名 .bak
    if exist "%%~fF.bak" del /q "%%~fF.bak"
    ren "%%~fF" "%%~nxF.bak"
    REM 临时文件改名为目标 (去掉 _converting 后缀, 强制 .mp4)
    ren "%%~dpnF_converting.mp4" "%%~nF.mp4"
    echo     [完成] → %%~nF.mp4
    set /a SUCCESSED+=1
  ) else (
    echo     [失败] ffmpeg 报错, 请检查视频文件
  )
  echo.

  :continue
)

echo ============================================================
echo   转码完成!
echo   扫描: !COUNT! 个文件
echo   成功: !SUCCESSED! 个
echo.
echo   原文件已备份为 .bak
echo   确认转码效果后删除备份:
echo     del /s /q "data\videos\*.bak"
echo ============================================================
pause
