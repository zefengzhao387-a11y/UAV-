/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["onnxruntime-web"],
  async rewrites() {
    return [{ source: "/favicon.ico", destination: "/favicon.svg" }];
  }
};

module.exports = nextConfig;
