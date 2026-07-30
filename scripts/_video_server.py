"""轻量视频静态服务 (端口 9802).

独立于 Next.js, 避免 next dev JIT 编译与视频流抢 CPU/IO.
支持 Range 请求 (seek), 中文路径, CORS.

用法: .venv/Scripts/python.exe scripts/_video_server.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("video_server")

HOST = "127.0.0.1"
PORT = 9802

# root = 项目根目录/data/videos
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos")

VIDEO_EXTS = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}

CHUNK = 256 * 1024  # 256KB per read


class VideoHandler(BaseHTTPRequestHandler):
    server_version = "VideoLensVideo/1.0"

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_HEAD(self):
        self._serve(head_only=True)

    def do_GET(self):
        self._serve(head_only=False)

    def _serve(self, head_only: bool):
        # URL 解码 (中文路径)
        raw_path = urllib.parse.unquote(self.path.split("?")[0].lstrip("/"))
        # 安全: 拒绝 .. 遍历
        if ".." in raw_path:
            self.send_error(403, "Forbidden")
            return

        # 尝试匹配视频文件 (支持无扩展名 / 带扩展名)
        file_path = None
        # 先试 raw_path 本身 (可能带扩展名)
        candidate = os.path.join(ROOT, raw_path.replace("/", os.sep))
        if os.path.isfile(candidate):
            ext = os.path.splitext(candidate)[1].lower()
            if ext in VIDEO_EXTS:
                file_path = candidate
        # 再试补扩展名
        if file_path is None:
            for ext in VIDEO_EXTS:
                candidate = os.path.join(ROOT, raw_path.replace("/", os.sep) + ext)
                if os.path.isfile(candidate):
                    file_path = candidate
                    break

        if file_path is None:
            self.send_error(404, "Not Found")
            return

        file_size = os.path.getsize(file_path)
        content_type = VIDEO_EXTS.get(os.path.splitext(file_path)[1].lower(), "application/octet-stream")

        # Range 解析
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        is_partial = False

        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                is_partial = True
                if start >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self._cors_headers()
                    self.end_headers()
                    return

        chunk_size = end - start + 1

        # 响应头
        if is_partial:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(chunk_size))
        self.send_header("Accept-Ranges", "bytes")
        self._cors_headers()
        self.end_headers()

        if head_only:
            return

        # 流式输出
        try:
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read = min(CHUNK, remaining)
                    data = f.read(read)
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # 客户端 seek/断开, 正常

    def log_message(self, fmt, *args):
        if self.command in ("GET", "HEAD"):
            logger.info(f"{self.command} {self.path} → {args[1]}")


def main():
    if not os.path.isdir(ROOT):
        logger.error(f"视频目录不存在: {ROOT}")
        sys.exit(1)
    server = ThreadingHTTPServer((HOST, PORT), VideoHandler)
    logger.info(f"视频服务启动: http://{HOST}:{PORT}  root={ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("停止")
        server.shutdown()


if __name__ == "__main__":
    main()
