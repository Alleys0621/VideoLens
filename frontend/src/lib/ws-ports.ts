/**
 * WS / 直连静态服务端口派生.
 *
 * 三场景自动适配, 不依赖 env:
 *  - 协作者本机 (localhost) / 局域网 (RFC1918 内网 IP): 服务原端口 9800/9801/9802
 *  - 公网入口 (西电广研院申请到的 TCP 映射 218.19.14.198): 7070/7071/7072
 *
 * 公网映射端口为固定申请结果, 不在 RFC1918 段内, 所以用 hostname 是否
 * 命中内网段来判定走哪组端口.
 */

export type WsService = "asr" | "tts" | "video";

const LAN_PORTS: Record<WsService, number> = {
  asr: 9800,
  tts: 9801,
  video: 9802,
};

const PUBLIC_PORTS: Record<WsService, number> = {
  asr: 7070,
  tts: 7071,
  video: 7072,
};

/** hostname 是否属于本机/局域网 (RFC1918 + loopback + mDNS). */
function isPrivateHost(hostname: string): boolean {
  if (!hostname) return true;
  if (hostname === "localhost") return true;
  if (hostname.endsWith(".local")) return true;
  if (hostname === "::1") return true;
  if (/^127\./.test(hostname)) return true;
  if (/^10\./.test(hostname)) return true;
  if (/^192\.168\./.test(hostname)) return true;
  const m = hostname.match(/^172\.(\d+)\./);
  if (m && +m[1] >= 16 && +m[1] <= 31) return true;
  return false;
}

/** 按 service 和当前 hostname 返回对应端口. SSR 安全 (无 window 时走 LAN 默认). */
export function resolveWsPort(service: WsService): number {
  if (typeof window === "undefined") return LAN_PORTS[service];
  return isPrivateHost(window.location.hostname)
    ? LAN_PORTS[service]
    : PUBLIC_PORTS[service];
}
