import { NextRequest, NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";

export const dynamic = "force-dynamic";

/** webm → wav (16kHz/mono/16bit) via ffmpeg pipe */
function webmToWav(webm: Buffer): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const ff = spawn("ffmpeg", [
      "-i", "pipe:0",
      "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
      "-f", "wav", "pipe:1",
    ]);
    const chunks: Buffer[] = [];
    ff.stdout.on("data", (c) => chunks.push(c));
    ff.on("close", (code) =>
      code === 0
        ? resolve(Buffer.concat(chunks))
        : reject(new Error(`ffmpeg exit ${code}`)),
    );
    ff.on("error", reject);
    ff.stdin.write(webm);
    ff.stdin.end();
  });
}

/** ASR: 音频 blob → Python asr.py (paraformer + omni fallback) → 文字 */
export async function POST(req: NextRequest): Promise<Response> {
  const t0 = Date.now();
  const formData = await req.formData();
  const audioFile = formData.get("audio") as File;
  if (!audioFile) {
    return NextResponse.json({ error: "no audio" }, { status: 400 });
  }

  // webm → wav
  const webmBuf = Buffer.from(await audioFile.arrayBuffer());
  let wavBuf: Buffer;
  try {
    wavBuf = await webmToWav(webmBuf);
  } catch (e) {
    return NextResponse.json({ error: `ffmpeg: ${e}` }, { status: 500 });
  }

  // child_process 调 Python asr.py (stdin wav → stdout text)
  const projectRoot = path.resolve(process.cwd(), "..");
  const python = path.join(projectRoot, ".venv", "Scripts", "python.exe");

  return new Promise<Response>((resolve) => {
    const py = spawn(python, ["-m", "src.agent.asr"], {
      cwd: projectRoot,
    });
    const outChunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    py.stdout.on("data", (c) => outChunks.push(c));
    py.stderr.on("data", (c) => errChunks.push(c));
    py.on("close", (code) => {
      const ms = Date.now() - t0;
      if (code === 0) {
        const text = Buffer.concat(outChunks).toString("utf-8").trim();
        resolve(NextResponse.json({ text, ms }));
      } else {
        const err = Buffer.concat(errChunks).toString("utf-8").slice(0, 200);
        resolve(
          NextResponse.json(
            { error: `asr exit ${code}: ${err}`, ms },
            { status: 500 },
          ),
        );
      }
    });
    py.on("error", (e) => {
      resolve(
        NextResponse.json({ error: `spawn: ${e.message}` }, { status: 500 }),
      );
    });
    py.stdin.write(wavBuf);
    py.stdin.end();
  });
}
