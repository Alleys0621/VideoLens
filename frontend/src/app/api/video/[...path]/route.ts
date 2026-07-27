import { NextRequest } from "next/server";
import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";

// force-dynamic: 视频流不能被缓存 (seek 的 Range 请求每次不同, 缓存会导致返回错误块)
export const dynamic = "force-dynamic";

const VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi"];

function getContentType(fp: string): string {
  const ext = path.extname(fp).toLowerCase();
  return (
    {
      ".mp4": "video/mp4",
      ".mkv": "video/x-matroska",
      ".mov": "video/quicktime",
      ".avi": "video/x-msvideo",
    }[ext] ?? "application/octet-stream"
  );
}

function resolveVideoFile(videoDir: string): string | null {
  const videosRoot = path.resolve(process.cwd(), "..", "data", "videos");
  for (const ext of VIDEO_EXTS) {
    const p = path.join(videosRoot, `${videoDir}${ext}`);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/**
 * 把 Node 的 fs 读流转成 Web ReadableStream 用于 Response body.
 *
 * 关键: 客户端 seek/断开会让 Web stream 被 cancel, 此时如果 fs stream 还在推数据,
 * 写到一个已关闭的 controller 会抛 "Controller is already closed" (uncaughtException).
 * 这里手动桥接 + 在 cancel/error 时 destroy fs stream, 并吞掉 controller 已关闭后的写入错误.
 */
function nodeStreamToWebResponseStream(
  nodeStream: Readable,
  status: number,
  headers: Record<string, string>,
): Response {
  const webStream = new ReadableStream<Uint8Array>({
    start(controller) {
      nodeStream.on("data", (chunk: Buffer) => {
        const desired = controller.desiredSize;
        if (desired !== null && desired <= 0) {
          // 简易背压: 暂停, 等下游拉取
          nodeStream.pause();
        }
        try {
          controller.enqueue(chunk);
        } catch {
          // controller 已关闭 (客户端断开), 忽略
        }
        const after = controller.desiredSize;
        if (nodeStream.isPaused() && after !== null && after > 0) {
          nodeStream.resume();
        }
      });
      nodeStream.on("end", () => {
        try {
          controller.close();
        } catch {
          // 已关闭
        }
      });
      nodeStream.on("error", (err) => {
        console.error("[video/stream] fs error:", err);
        try {
          controller.error(err);
        } catch {
          // 已关闭
        }
        nodeStream.destroy();
      });
    },
    cancel() {
      // 客户端取消 (seek/关页面) — 立刻停掉 fs 读流, 避免后续 enqueue 到已关闭 controller
      nodeStream.destroy();
    },
  });

  return new Response(webStream, { status, headers });
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path: pathParts } = await params;
  const videoDir = pathParts.join("/");
  const filePath = resolveVideoFile(videoDir);

  if (!filePath) {
    return new Response(`Video not found: ${videoDir}`, { status: 404 });
  }

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const range = req.headers.get("range");
  const contentType = getContentType(filePath);

  if (range) {
    // 解析 Range: bytes=start-end
    // 浏览器请求 bytes=X- 时, 返回 X 到文件末尾 (不限 chunk, 减少往返)
    const match = /bytes=(\d+)-(\d*)/.exec(range);
    if (match) {
      const start = parseInt(match[1], 10);
      const end = match[2] ? parseInt(match[2], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      const nodeStream = fs.createReadStream(filePath, { start, end });

      return nodeStreamToWebResponseStream(nodeStream, 206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunkSize.toString(),
        "Content-Type": contentType,
      });
    }
  }

  // 无 Range, 返回完整文件
  const nodeStream = fs.createReadStream(filePath);
  return nodeStreamToWebResponseStream(nodeStream, 200, {
    "Content-Length": fileSize.toString(),
    "Content-Type": contentType,
    "Accept-Ranges": "bytes",
  });
}
