# VideoLens 批量 HLS 转码脚本
# 用法: 在 D:\Projects\VideoLens 目录下 powershell -ExecutionPolicy Bypass -File scripts\transcode_hls.ps1
#
# 转码目标:
#   - 视频编码: H.264 (libx264), CRF 24, maxrate 1.8 Mbps (适配腾讯云 3M 带宽)
#   - 音频编码: AAC 96kbps 立体声
#   - 容器: HLS (.m3u8 + .ts), 10 秒切片
#   - 分辨率: 保持原始 (4:3 1440x1080 不动)
#
# 转码后输出路径与 /api/hls/[...path] 路由对齐:
#   data/hls/{作品}/{季}/{集}/playlist.m3u8 + seg-001.ts ...

# 要转码的视频列表 (按用户需求)
$videos = @(
    @{ src = "data/videos/家有儿女/第一季/第01集.mkv"; out = "data/hls/家有儿女/第一季/第01集" },
    @{ src = "data/videos/家有儿女/第一季/第02集.mkv"; out = "data/hls/家有儿女/第一季/第02集" },
    @{ src = "data/videos/家有儿女/第一季/第03集.mkv"; out = "data/hls/家有儿女/第一季/第03集" },
    @{ src = "data/videos/家有儿女/第二季/第001集.mp4"; out = "data/hls/家有儿女/第二季/第001集" }
)

# 切到脚本所在仓库根目录
Set-Location -Path (Resolve-Path "$PSScriptRoot/..")

$success = 0
$failed = @()

foreach ($v in $videos) {
    $src = $v.src
    $outDir = $v.out

    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "Source: $src"
    Write-Host "Target: $outDir/playlist.m3u8"
    Write-Host "=============================================" -ForegroundColor Cyan

    if (-not (Test-Path $src)) {
        Write-Host "[SKIP] Source not found: $src" -ForegroundColor Yellow
        $failed += $src
        continue
    }

    # 创建输出目录
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    # ffmpeg 参数说明:
    #   -preset fast        : 编码速度优先 (转码快, 文件稍大)
    #   -crf 24              : 质量档位 (18=无损, 23=默认, 28=差, 24 是质量/大小平衡点)
    #   -maxrate 1800k       : 视频峰值码率 1.8 Mbps (3M 带宽留余量给音频)
    #   -bufsize 3600k       : 码率缓冲 (2x maxrate, 控制码率波动)
    #   -pix_fmt yuv420p     : Safari/iOS 强制要求
    #   -ac 2                : 双声道立体声 (5.1 → 2.0 兼容)
    #   -movflags +faststart : MP4 元数据前置 (HLS 不强需要, 但加上无害)
    #   -hls_time 10         : 每段 10 秒
    #   -hls_playlist_type vod : VOD 模式 (不是直播 sliding window)
    #   -hls_segment_type mpegts : .ts 容器 (浏览器原生兼容)
    $args = @(
        "-y", "-i", $src,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "24",
        "-maxrate", "1800k",
        "-bufsize", "3600k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "96k",
        "-ac", "2",
        "-hls_time", "10",
        "-hls_playlist_type", "vod",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", "$outDir/seg-%03d.ts",
        "$outDir/playlist.m3u8"
    )

    $t0 = Get-Date
    & ffmpeg @args
    $elapsed = (Get-Date) - $t0

    if ($LASTEXITCODE -eq 0) {
        $size = (Get-ChildItem $outDir | Measure-Object Length -Sum).Sum / 1MB
        Write-Host ("[OK] Done in {0:F1} min, output {1:F1} MB" -f $elapsed.TotalMinutes, $size) -ForegroundColor Green
        $success++
    } else {
        Write-Host "[FAIL] ffmpeg exit $LASTEXITCODE" -ForegroundColor Red
        $failed += $src
    }
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Summary: $success ok, $($failed.Count) failed"
if ($failed.Count -gt 0) {
    Write-Host "Failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" }
}
Write-Host "=============================================" -ForegroundColor Cyan
