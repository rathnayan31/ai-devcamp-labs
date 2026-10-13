"""AG-UI bridge: exposes the social_poster agent to the CopilotKit frontend.

Run from the repo root:  uv run uvicorn backend.main:app --port 8000
"""

import logging
import pathlib

from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv(pathlib.Path(__file__).parent / "social_poster" / ".env")

from backend.social_poster import db  # noqa: E402  (needs .env first)
from backend.social_poster.agent import root_agent  # noqa: E402  (needs .env first)
from backend.social_poster.runtime import resolve_agent_engine  # noqa: E402

log = logging.getLogger("social_spark")

# Lab 2: on Agent Runtime the container scales to zero, so conversations
# (Sessions) and cross-conversation facts (Memory Bank) live on the engine.
# Locally, None keeps ADKAgent's in-memory defaults.
_engine = resolve_agent_engine()
if _engine:
    from google.adk.memory import VertexAiMemoryBankService
    from google.adk.sessions import VertexAiSessionService

    _engine_kwargs = {
        "project": _engine["project"],
        "location": _engine["location"],
        "agent_engine_id": _engine["engine"],
    }
    session_service = VertexAiSessionService(**_engine_kwargs)
    memory_service = VertexAiMemoryBankService(**_engine_kwargs)
    log.warning("Managed Sessions + Memory Bank on engine %s (%s)", _engine["engine"], _engine["location"])
else:
    session_service = memory_service = None

app = FastAPI(title="social-agent AG-UI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

adk_agent = ADKAgent(
    adk_agent=root_agent,
    app_name="social_poster",
    user_id="devcamp-user",  # single-user POC; extract from auth in real apps
    session_service=session_service,
    memory_service=memory_service,
    # ADKAgent's idle cleanup deletes sessions by default; managed sessions
    # must outlive this process, so only delete the local in-memory ones.
    delete_session_on_cleanup=_engine is None,
)

add_adk_fastapi_endpoint(app, adk_agent, path="/api/adk")

# Serve generate_image's output files so the frontend's image gallery
# (PostGallery.tsx) can render them — tools.py writes local file paths, which
# the browser has no way to read directly otherwise.
GALLERY_DIR = pathlib.Path(__file__).resolve().parents[1] / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=GALLERY_DIR), name="outputs")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/posts")
def posts() -> dict:
    """Published posts (with whatever image they used), read from the local
    SQLite store — durable across reloads/restarts, unlike ADK session state
    or the browser's own chat history."""
    return {"posts": db.list_posts()}
