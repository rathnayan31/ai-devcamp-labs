"""Configure the deployed engine's Memory Bank (Lab 2, Step 5).

    GOOGLE_CLOUD_PROJECT=YOUR_PROJECT AGENT_ENGINE_ID=ENGINE_ID \
        uv run python backend/memory_bank_config.py apply
    ... uv run python backend/memory_bank_config.py show

``apply`` is a PATCH of the engine's ``context_spec`` only (update mask
``context_spec``): no redeploy, and the container, env vars and identity are
untouched. Deleting the engine deletes its Memory Bank too.

What gets remembered is decided by topics: the four managed ones, plus a
custom ``posting_style`` topic for how this user likes their posts written.
Memories expire after ``MEMORY_TTL_DAYS`` (default 90).
"""

from __future__ import annotations

import json
import os
import sys

import vertexai

LOCATION = os.environ.get("AGENT_ENGINE_LOCATION", "us-central1")
TTL_DAYS = int(os.environ.get("MEMORY_TTL_DAYS", "90"))

MANAGED_TOPICS = ("USER_PREFERENCES", "EXPLICIT_INSTRUCTIONS", "USER_PERSONAL_INFO", "KEY_CONVERSATION_DETAILS")

POSTING_STYLE = {
    "label": "posting_style",
    "description": (
        "How the user wants their social posts written: tone and voice, length, "
        "hashtag and emoji use, calls to action, preferred platforms (LinkedIn, X), "
        "posting times, and topics or phrasings they've asked to avoid. Also any "
        "edits they made to a draft before approving it, as a standing preference."
    ),
}


def memory_bank_config() -> dict:
    return {
        "customization_configs": [
            {
                "memory_topics": [
                    *({"managed_memory_topic": {"managed_topic_enum": t}} for t in MANAGED_TOPICS),
                    {"custom_memory_topic": POSTING_STYLE},
                ],
            }
        ],
        "ttl_config": {"default_ttl": f"{TTL_DAYS * 24 * 3600}s"},
    }


def _engine_name() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    engine = os.environ.get("AGENT_ENGINE_ID", "")
    if not project or not engine:
        sys.exit("Set GOOGLE_CLOUD_PROJECT and AGENT_ENGINE_ID (see `agents-cli deploy --list`).")
    return f"projects/{project}/locations/{LOCATION}/reasoningEngines/{engine}"


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ("apply", "show"):
        sys.exit("usage: memory_bank_config.py apply|show")
    name = _engine_name()
    client = vertexai.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=LOCATION)
    if action == "apply":
        client.agent_engines.update(
            name=name, config={"context_spec": {"memory_bank_config": memory_bank_config()}}
        )
        print(f"Applied Memory Bank config to {name}")
    engine = client.agent_engines.get(name=name)
    spec = engine.api_resource.context_spec
    print(json.dumps(spec.model_dump(mode="json", exclude_none=True) if spec else {}, indent=2))


if __name__ == "__main__":
    main()
