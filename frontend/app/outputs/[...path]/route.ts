import { NextRequest } from "next/server";
import { proxyToBackend } from "../../lib/backend";

// generate_image's files, served by the backend's /outputs mount.
export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyToBackend(req, `/outputs/${path.map(encodeURIComponent).join("/")}`);
}
