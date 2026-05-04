/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/favicon.ico", destination: "/favicon.svg" }];
  }
};

module.exports = nextConfig;
