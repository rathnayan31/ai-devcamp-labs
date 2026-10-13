import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// Bridges the browser to the ADK backend's AG-UI endpoint, via this app's own
// /api/adk proxy (app/api/adk/route.ts), which attaches a Google token when
// ADK_BACKEND_ORIGIN points at Agent Runtime. Don't give HttpAgent a custom
// fetch that adds the token instead: it works in `next dev` but the header is
// dropped in a production build (401 UNAUTHENTICATED).
export const POST = async (req: NextRequest) => {
  // Call ourselves over loopback: the public host may be a proxy (e.g.
  // `gcloud run services proxy`) that isn't reachable from inside the container.
  const port = process.env.PORT || req.nextUrl.port || "3000";
  const runtime = new CopilotRuntime({
    agents: {
      social_poster: new HttpAgent({ url: `http://127.0.0.1:${port}/api/adk` }),
    },
  });
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
