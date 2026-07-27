/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  // 允许通过 Cloudflare quick tunnel 域名访问 dev server 时加载 /_next/* 资源
  allowedDevOrigins: ["*.trycloudflare.com"],
};

export default nextConfig;
