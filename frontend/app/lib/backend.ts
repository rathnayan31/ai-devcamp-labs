import { GoogleAuth } from "google-auth-library";

// Where the ADK backend lives. Unset: the local `./run-backend.sh`. Set it to
// the deployed engine's passthrough base (…/reasoningEngines/v1/projects/…/api)
// to talk to Agent Runtime instead. Server-side only: never sent to the browser.
const ORIGIN = (process.env.ADK_BACKEND_ORIGIN || "http://localhost:8000").replace(/\/+$/, "");

const auth = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });

// Only Google endpoints get a token: a local backend needs none, and an ADC
// token must never be sent to any other host.
const isGoogle = new URL(ORIGIN).hostname.endsWith(".googleapis.com");

async function authHeader(): Promise<Record<string, string>> {
  if (!isGoogle) return {};
  // ADC: your gcloud login locally, the frontend service account on Cloud Run.
  const token = await auth.getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Forward a request to the backend and pass the response straight through.
 * The body is streamed, not buffered, so AG-UI's SSE keeps streaming.
 */
export async function proxyToBackend(req: Request, path: string): Promise<Response> {
  const headers: Record<string, string> = { ...(await authHeader()) };
  for (const name of ["content-type", "accept"]) {
    const value = req.headers.get(name);
    if (value) headers[name] = value;
  }
  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  const upstream = await fetch(`${ORIGIN}${path}`, {
    method: req.method,
    headers,
    body: hasBody ? req.body : undefined,
    // Required by Node's fetch to stream a request body.
    ...(hasBody ? { duplex: "half" } : {}),
    cache: "no-store",
  } as RequestInit);
  const out = new Headers();
  for (const name of ["content-type", "cache-control", "content-length"]) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}
