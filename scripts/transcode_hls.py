"""VideoLens 批量 HLS 转码脚本.

转码目标:
  - 视频编码: H.264 (libx264), CRF 24, maxrate 1.8 Mbps (适配腾讯云 3M 带宽)
  - 音频编码: AAC 96kbps 立体声
  - 容器: HLS (.m3u8 + .ts), 10 秒切片
  - 分辨率: 保持原始 (4:3 1440x1080 不动)

用法:
  python scripts/transcode_hls.py
"""

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 要转码的视频列表 (源是相对项目根的路径, 输出对齐 /api/hls/[...path] 路由)
VIDEOS = [
    ("data/videos/家有儿女/第一季/第01集.mkv", "data/hls/家有儿女/第一季/第01集"),
    ("data/videos/家有儿女/第一季/第02集.mkv", "data/hls/家有儿女/第一季/第02集"),
    ("data/videos/家有儿女/第一季/第03集.mkv", "data/hls/家有儿女/第一季/第03集"),
    ("data/videos/家有儿女/第二季/第001集.mp4", "data/hls/家有儿女/第二季/第001集"),
]


def transcode_one(src: Path, out_dir: Path) -> bool:
    """转码单个视频为 HLS. 返回 True 成功, False 失败."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg 参数说明:
    #   -preset fast        : 编码速度优先 (转码快, 文件稍大)
    #   -crf 24              : 质量档位 (18=无损, 23=默认, 28=差)
    #   -maxrate 1800k       : 视频峰值码率 1.8 Mbps (3M 带宽留余量给音频)
    #   -bufsize 3600k       : 码率缓冲 (2x maxrate)
    #   -pix_fmt yuv420p     : Safari/iOS 强制要求
    #   -ac 2                : 双声道立体声 (5.1 → 2.0 兼容)
    #   -hls_time 10         : 每段 10 秒
    #   -hls_playlist_type vod : VOD 模式 (不是直播 sliding window)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
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
        "-hls_segment_filename", str(out_dir / "seg-%03d.ts"),
        str(out_dir / "playlist.m3u8"),
    ]

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        total_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1024 / 1024
        print(f"[OK] Done in {elapsed / 60:.1f} min, output {total_mb:.1f} MB")
        return True
    else:
        print(f"[FAIL] ffmpeg exit {result.returncode}")
        return False


def main():
    success = 0
    failed = []

    for src_rel, out_rel in VIDEOS:
        src = ROOT / src_rel
        out_dir = ROOT / out_rel

        print()
        print("=" * 60)
        print(f"Source: {src_rel}")
        print(f"Target: {out_rel}/playlist.m3u8")
        print("=" * 60)

        if not src.exists():
            print(f"[SKIP] Source not found: {src_rel}")
            failed.append(src_rel)
            continue

        if transcode_one(src, out_dir):
            success += 1
        else:
            failed.append(src_rel)

    print()
    print("=" * 60)
    print(f"Summary: {success} ok, {len(failed)} failed")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
