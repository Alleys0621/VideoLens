import { NextRequest, NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";

export const dynamic = "force-dynamic";

/** TTS: 文字 → 调 Python tts.py (CosyVoice) → base64 音频 */
export async function POST(req: NextRequest): Promise<Response> {
  const { text } = await req.json();
  if (!text) return NextResponse.json({ error: "no text" }, { status: 400 });

  const t0 = Date.now();
  const projectRoot = path.resolve(process.cwd(), "..");
  const python = path.join(projectRoot, ".venv", "Scripts", "python.exe");

  return new Promise((resolve) => {
    const py = spawn(python, ["-m", "src.agent.tts", text], {
      cwd: projectRoot,
    });
    const chunks: Buffer[] = [];
    py.stdout.on("data", (c) => chunks.push(c));
    py.on("close", (code) => {
      const ms = Date.now() - t0;
      if (code === 0 && chunks.length > 0) {
        const b64 = Buffer.concat(chunks).toString("utf-8").trim();
        resolve(NextResponse.json({ audio: b64, ms }));
      } else {
        resolve(
          NextResponse.json({ error: `tts exit ${code}`, ms }, { status: 500 }),
        );
      }
    });
    py.on("error", (e) => {
      resolve(NextResponse.json({ error: `spawn: ${e.message}` }, { status: 500 }));
    });
  });
}
