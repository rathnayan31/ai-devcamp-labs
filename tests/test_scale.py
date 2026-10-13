"""Lab 2 (Scale) wiring, offline: engine detection and saving to Memory Bank."""

import asyncio
from types import SimpleNamespace

from backend.social_poster import agent
from backend.social_poster.runtime import resolve_agent_engine

APP_URL = (
    "https://us-central1-aiplatform.googleapis.com/reasoningEngines/v1/"
    "projects/my-proj/locations/us-central1/reasoningEngines/1234567890/api"
)


def _clear_engine_env(monkeypatch):
    for name in ("APP_URL", "GOOGLE_CLOUD_AGENT_ENGINE_ID", "AGENT_ENGINE_LOCATION"):
        monkeypatch.delenv(name, raising=False)


def test_local_run_has_no_engine(monkeypatch):
    _clear_engine_env(monkeypatch)
    assert resolve_agent_engine() is None


def test_engine_parsed_from_app_url_not_global_location(monkeypatch):
    _clear_engine_env(monkeypatch)
    monkeypatch.setenv("APP_URL", APP_URL)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "987654321")  # the number, in the container
    assert resolve_agent_engine() == {"project": "my-proj", "location": "us-central1", "engine": "1234567890"}


def test_first_deploy_falls_back_to_platform_engine_id(monkeypatch):
    _clear_engine_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "987654321")
    assert resolve_agent_engine() is None  # location unknown: don't guess
    monkeypatch.setenv("AGENT_ENGINE_LOCATION", "us-central1")
    assert resolve_agent_engine() == {"project": "987654321", "location": "us-central1", "engine": "1234567890"}


def test_load_memory_shows_as_consulting_memory():
    assert agent._TOOL_STAGES["load_memory"] == "consulting_memory"


class _Ctx:
    def __init__(self, state):
        self.state = state
        self.saved = 0

    async def add_session_to_memory(self):
        self.saved += 1


def _after(ctx):
    asyncio.run(agent._stage_after_agent(ctx))


def test_posted_session_saved_to_memory_once(monkeypatch):
    monkeypatch.setattr(agent, "memory_bank_enabled", lambda: True)
    ctx = _Ctx({"pipeline_stage": "posted"})
    _after(ctx)
    _after(ctx)  # a later turn in the same conversation
    assert ctx.saved == 1
    assert ctx.state[agent.MEMORY_INGESTED_KEY] is True


def test_unposted_or_local_sessions_not_saved(monkeypatch):
    monkeypatch.setattr(agent, "memory_bank_enabled", lambda: True)
    drafting = _Ctx({"pipeline_stage": "drafting"})
    _after(drafting)
    assert drafting.saved == 0 and drafting.state["pipeline_stage"] == "awaiting_approval"

    monkeypatch.setattr(agent, "memory_bank_enabled", lambda: False)
    local = _Ctx({"pipeline_stage": "posted"})
    _after(local)
    assert local.saved == 0


def test_memory_failure_never_blocks_the_turn(monkeypatch):
    monkeypatch.setattr(agent, "memory_bank_enabled", lambda: True)

    class Failing(_Ctx):
        async def add_session_to_memory(self):
            raise RuntimeError("memory bank down")

    ctx = Failing({"pipeline_stage": "posted"})
    _after(ctx)  # must not raise
    assert not ctx.state.get(agent.MEMORY_INGESTED_KEY)


def test_load_memory_tool_only_when_deployed():
    names = {getattr(t, "name", None) for t in agent.root_agent.tools}
    assert "load_memory" not in names  # this test process has no engine
