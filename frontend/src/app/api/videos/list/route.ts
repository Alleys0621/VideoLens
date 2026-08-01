import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";

export type Episode = { dir: string; label: string };
export type Season = { name: string; episodes: Episode[] };
export type Show = {
  name: string;
  seasons: Season[];
  directEpisodes: Episode[]; // 无季的直接集 (如 家有儿女/第001集)
};

/** 列出已建库视频, 树形结构: show → season → episodes */
export async function GET() {
  const outputRoot = path.resolve(process.cwd(), "..", "data", "output");
  if (!fs.existsSync(outputRoot)) return NextResponse.json([]);

  const shows: Show[] = [];

  for (const showName of fs.readdirSync(outputRoot)) {
    if (showName.startsWith("_") || showName.startsWith(".")) continue;
    const showDir = path.join(outputRoot, showName);
    if (!fs.statSync(showDir).isDirectory()) continue;

    const show: Show = { name: showName, seasons: [], directEpisodes: [] };

    for (const entry of fs.readdirSync(showDir)) {
      if (entry.startsWith(".")) continue;
      const entryPath = path.join(showDir, entry);
      if (!fs.statSync(entryPath).isDirectory()) continue;

      if (fs.existsSync(path.join(entryPath, "stage3_kb.json"))) {
        // 直接集 (show/episode, 无季)
        show.directEpisodes.push({
          dir: `${showName}/${entry}`,
          label: entry,
        });
      } else {
        // 季 (show/season/episode)
        const season: Season = { name: entry, episodes: [] };
        for (const ep of fs.readdirSync(entryPath)) {
          if (ep.startsWith(".")) continue;
          const epPath = path.join(entryPath, ep);
          if (
            fs.statSync(epPath).isDirectory() &&
            fs.existsSync(path.join(epPath, "stage3_kb.json"))
          ) {
            season.episodes.push({
              dir: `${showName}/${entry}/${ep}`,
              label: ep,
            });
          }
        }
        if (season.episodes.length > 0) show.seasons.push(season);
      }
    }

    if (show.directEpisodes.length > 0 || show.seasons.length > 0) {
      shows.push(show);
    }
  }

  return NextResponse.json(shows);
}
