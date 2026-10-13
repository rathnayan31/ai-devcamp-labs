"""Where the agent is running: locally, or on Agent Runtime (Lab 2: Scale).

Agent Runtime injects ``APP_URL``, the engine's own ``/api`` passthrough URL:

    https://us-central1-aiplatform.googleapis.com/reasoningEngines/v1/projects/P/locations/L/reasoningEngines/ID/api

Managed Sessions and Memory Bank are keyed on that engine, so everything is
parsed from it rather than configured by hand. Two env vars that look right
aren't: ``GOOGLE_CLOUD_LOCATION`` is ``global`` (for Gemini) while the engine
lives in ``us-central1``, and in the container ``GOOGLE_CLOUD_PROJECT`` is the
project *number*. agents-cli only adds ``APP_URL`` on an update, so a first
deploy falls back to the platform's ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` plus
``AGENT_ENGINE_LOCATION``.
"""

from __future__ import annotations

import os
import re

_APP_URL_RE = re.compile(
    r"/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/reasoningEngines/(?P<engine>\d+)"
)


def resolve_agent_engine() -> dict[str, str] | None:
    """{project, location, engine} on Agent Runtime, else None (local run)."""
    match = _APP_URL_RE.search(os.environ.get("APP_URL", ""))
    if match:
        return match.groupdict()
    engine = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "").strip().rsplit("/", 1)[-1]
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("AGENT_ENGINE_LOCATION", "")
    if engine and project and location:
        return {"project": project, "location": location, "engine": engine}
    return None


def memory_bank_enabled() -> bool:
    """Memory Bank needs an engine, so it's on only when deployed."""
    return resolve_agent_engine() is not None
