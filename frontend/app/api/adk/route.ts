import { NextRequest } from "next/server";
import { proxyToBackend } from "../../lib/backend";

// Same-origin AG-UI endpoint. CopilotKit's HttpAgent posts here; the proxy
// attaches the Google token server-side when the backend is Agent Runtime.
export const POST = (req: NextRequest) => proxyToBackend(req, "/api/adk");
