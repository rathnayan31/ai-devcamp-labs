import { NextRequest } from "next/server";
import { proxyToBackend } from "../../lib/backend";

// Published-post history for PostGallery, read from the backend's SQLite store.
export const GET = (req: NextRequest) => proxyToBackend(req, "/api/posts");
