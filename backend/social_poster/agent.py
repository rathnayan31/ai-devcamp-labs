import os
import pathlib
from google.adk.agents import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from .tools import (
    calculate_dasha_periods,
    generate_image,
    search_remedy_knowledge_base,
)
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
import sys
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Any, Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.load_memory_tool import load_memory_tool
import logging

from .runtime import memory_bank_enabled

log = logging.getLogger(__name__)

SKILLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills"
BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MCP_SERVER = PROJECT_ROOT / "mcp" / "linkedin_server.py"

buffer_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.buffer.com/mcp",
        headers={"Authorization": f"Bearer {BUFFER_API_KEY}"},
    ),
    # Framework-level approval gate: see below, not just an instruction.
    require_confirmation=True,
)

linkedin_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER)],
            env={
                "DRY_RUN": os.environ.get("DRY_RUN", "true"),
                "LINKEDIN_ACCESS_TOKEN": os.environ.get("LINKEDIN_ACCESS_TOKEN", ""),
            },
        ),
    ),
    tool_filter=["get_profile", "create_post"],
    # Buffer's server also has a create_post; Gemini rejects two functions with
    # the same name. The prefix renames these to linkedin_get_profile and
    # linkedin_create_post. The filter above still uses the original names.
    tool_name_prefix="linkedin",
    require_confirmation=True,
)
# POST_VIA overrides the LLM's own routing: "buffer" drops the LinkedIn
# tool entirely, "linkedin" drops Buffer even if a key is set. Blank
# exposes both and lets the agent route by what the user asks for.
POST_VIA = os.environ.get("POST_VIA", "").strip().lower()
use_linkedin = POST_VIA != "buffer"
use_buffer = bool(BUFFER_API_KEY) and POST_VIA != "linkedin"

dasha_skill = load_skill_from_dir(SKILLS_DIR / "dasha-timing")
remedy_skill = load_skill_from_dir(SKILLS_DIR / "remedy-kb-search")

dasha_skill_toolset = SkillToolset(skills=[dasha_skill])
remedy_skill_toolset = SkillToolset(skills=[remedy_skill])

skill_toolset = SkillToolset(
    skills=[
        load_skill_from_dir(SKILLS_DIR / "post-formatter"),
        load_skill_from_dir(SKILLS_DIR / "platform-style"),
        load_skill_from_dir(SKILLS_DIR / "brand-voice"),
        load_skill_from_dir(SKILLS_DIR / "poster-style"),
        dasha_skill,
        remedy_skill,
    ],
)

search_agent = Agent(
    name="web_researcher",
    model=os.environ.get("RESEARCH_MODEL", "gemini-flash-latest"),
    description="Researches facts, dates, and context on the web.",
    instruction="Research the given query and return a concise summary of relevant facts.",
    tools=[google_search],
)

dasha_timing_agent = Agent(
    name="dasha_timing_agent",
    model=os.environ.get("DASHATIMING_MODEL", "gemini-flash-latest"),
    description="Calculates and analyzes Vedic astrology Dasha timing cycles (Vimshottari Mahadasha, Antardasha) and planetary periods to identify timing windows.",
    instruction="""You are an expert Vedic astrologer specializing in Dasha timing analysis.
Your role is to analyze planetary periods (Mahadasha, Antardasha, Pratyantardasha) and transits.

Workflow:
1. When birth details or query dates are provided, call calculate_dasha_periods to retrieve structured timing facts.
2. Load and follow the dasha-timing skill for period interpretation and planetary significations.
3. Deliver a concise, objective analysis of the active planetary cycle, key life themes, and favorable versus challenging timeframes.""",
    tools=[calculate_dasha_periods, dasha_skill_toolset],
)
DashaTimingAgent = dasha_timing_agent

remedy_agent = Agent(
    name="remedy_agent",
    model=os.environ.get("REMEDY_MODEL", "gemini-flash-latest"),
    description="Provides authentic, grounded Vedic astrological remedies (mantras, gemstones with safety warnings, charitable giving, lifestyle upayas) based on planetary afflictions.",
    instruction="""You are an expert Vedic astrology remedy consultant.
Your role is to provide grounded, responsible astrological remedies (upayas) for planetary afflictions and doshas.

Workflow:
1. Search the verified remedy knowledge base using search_remedy_knowledge_base to retrieve grounded recommendations.
2. Load and follow the remedy-kb-search skill.
3. Recommend safe, sattvic remedies first (mantras, charitable acts/daan, lifestyle adjustments).
4. For gemstones, explicitly state safety cautions, ascendant dependency, and recommend consulting a qualified astrologer before wearing.
5. Never provide fear-based, superstitious, or commercialized claims.""",
    tools=[search_remedy_knowledge_base, remedy_skill_toolset],
)
RemedyAgent = remedy_agent

# idea -> consulting_memory -> researching -> analyzing_timing -> consulting_remedies
#      -> drafting -> awaiting_approval -> posted
_TOOL_STAGES = {
    "load_memory": "consulting_memory",
    "web_researcher": "researching",
    "dasha_timing_agent": "analyzing_timing",
    "remedy_agent": "consulting_remedies",
    "load_skill": "drafting",
    "generate_image": "drafting",
}

def _init_stage(callback_context: CallbackContext) -> None:
    if "pipeline_stage" not in callback_context.state:
        callback_context.state["pipeline_stage"] = "idea"


def _track_stage(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict
) -> Optional[dict]:
    if tool.name in _TOOL_STAGES:
        tool_context.state["pipeline_stage"] = _TOOL_STAGES[tool.name]
    elif tool.name in ("create_post", "linkedin_create_post") and tool_response.get("isError") is False:
        tool_context.state["pipeline_stage"] = "posted"  # only a completed MCP call
    return None  # never modify the tool result

_MID_PIPELINE = ("consulting_memory", "researching", "analyzing_timing", "consulting_remedies", "drafting")
MEMORY_INGESTED_KEY = "memory_ingested"


async def _stage_after_agent(callback_context: CallbackContext) -> None:
    # A turn that ends mid-pipeline is waiting on the user (draft approval).
    if callback_context.state.get("pipeline_stage") in _MID_PIPELINE:
        callback_context.state["pipeline_stage"] = "awaiting_approval"

    # Lab 2: save to Memory Bank once a post is published, once per
    # conversation. Nothing is saved automatically: the idle-session save
    # never fires on a container that scales to zero first. Must be awaited;
    # an un-awaited coroutine silently does nothing, and try/except won't see it.
    if (
        memory_bank_enabled()
        and callback_context.state.get("pipeline_stage") == "posted"
        and not callback_context.state.get(MEMORY_INGESTED_KEY)
    ):
        try:
            await callback_context.add_session_to_memory()
            callback_context.state[MEMORY_INGESTED_KEY] = True
        except Exception:  # memory is an enhancement, never a gate on posting
            log.exception("Could not save this session to Memory Bank")


root_agent = Agent(
    name="social_poster",
    model=os.environ.get("ORCHESTRATOR_MODEL", "gemini-flash-latest"),
    description="Helps the user turn an idea into a polished social media post, with specialist support for research, astrological timing, and grounded remedies.",
    instruction="""You help the user turn an idea into a social media post, coordinating specialist agents as needed.

Workflow:
0. If the load_memory tool is available, call it before drafting to recall
   this user's posting style and preferences from earlier conversations
   (tone, length, hashtags, emoji, platforms), and apply them unprompted.
   Anything the user says in this conversation overrides a memory.
1. If the idea needs facts, dates, or context, research it with the
   web_researcher tool first.
2. If the user asks about astrological timing, planetary cycles, or Vimshottari
   Dasha periods, consult the dasha_timing_agent tool.
3. If the user asks about astrological remedies, gemstones, mantras, or upayas,
   consult the remedy_agent tool.
4. When drafting, load and follow the relevant skills: post-formatter
   (structure), platform-style (rules for the target platform), and
   brand-voice (tone).
5. If the user asks for an image, load and follow the poster-style skill,
   then call generate_image with a detailed visual description and tell
   them where the file was saved.""",
    tools=[
        AgentTool(agent=search_agent),
        AgentTool(agent=dasha_timing_agent),
        AgentTool(agent=remedy_agent),
        generate_image,
        skill_toolset,
        linkedin_toolset,
        buffer_toolset,
        # Lab 2: Memory Bank only exists on Agent Runtime.
        *([load_memory_tool] if memory_bank_enabled() else []),
    ],
    before_agent_callback=_init_stage,
    after_tool_callback=_track_stage,
    after_agent_callback=_stage_after_agent,
)

__all__ = [
    "root_agent",
    "search_agent",
    "dasha_timing_agent",
    "DashaTimingAgent",
    "remedy_agent",
    "RemedyAgent",
]