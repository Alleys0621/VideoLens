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
      const webStream = Readable.toWeb(
        nodeStream,
      ) as ReadableStream<Uint8Array>;

      return new Response(webStream, {
        status: 206,
        headers: {
          "Content-Range": `bytes ${start}-${end}/${fileSize}`,
          "Accept-Ranges": "bytes",
          "Content-Length": chunkSize.toString(),
          "Content-Type": contentType,
        },
      });
    }
  }

  // 无 Range, 返回完整文件
  const nodeStream = fs.createReadStream(filePath);
  const webStream = Readable.toWeb(nodeStream) as ReadableStream<Uint8Array>;
  return new Response(webStream, {
    status: 200,
    headers: {
      "Content-Length": fileSize.toString(),
      "Content-Type": contentType,
      "Accept-Ranges": "bytes",
    },
  });
}
