"""
ZoneScore — Chat endpoint with tool-calling loop.

POST /api/chat/
Body: {"message": "...", "session_id": "optional"}
Returns: {"reply": "...", "zones": [...], "profile": {...}}

Orchestrates: LLM -> tool call -> search_zones -> LLM text reply.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .criteria_config import CRITERIA_CONFIG, CRITERION_IDS, DEFAULT_WEIGHT
from .llm_client import LLMClient, LLMResponse
from .ranking import search_zones, RankedZone
from .system_prompt import build_system_prompt

# ─── Tool definitions ───────────────────────────────────────────────────────
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_zones",
            "description": "Search and rank Tunisian zones by weighted criteria, location, and hard minimums.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_filter": {
                        "type": "string",
                        "description": "Governorate or delegation name (e.g. 'La Marsa', 'Tunis', 'Ariana').",
                    },
                    "weights": {
                        "type": "object",
                        "description": "Criterion importance levels: {criterion_id: 'low'|'medium'|'high'|'must_have'}",
                        "additionalProperties": {"type": "string", "enum": ["low", "medium", "high", "must_have"]},
                    },
                    "min_scores": {
                        "type": "object",
                        "description": "Hard minimum scores: {criterion_id: 0-100}",
                        "additionalProperties": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of zones to return (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "mentioned_criteria": {
                        "type": "array",
                        "items": {"type": "string", "enum": CRITERION_IDS},
                        "description": "Criterion ids the user actually mentioned.",
                    },
                },
                "required": ["weights", "mentioned_criteria"],
            },
        },
    }
]

# ─── In-memory session storage (Redis later) ────────────────────────────────
# {session_id: {"messages": [...], "profile": {...}, "turn": int}}
SESSIONS: OrderedDict[str, Dict[str, Any]] = OrderedDict()
MAX_SESSIONS = 1000
FOLLOW_UP_LIMIT = 3

# ─── Rate limiting (simple in-memory, per session) ──────────────────────────
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX = 30      # requests per window per session
_rate_logs: OrderedDict[str, List[float]] = OrderedDict()


def _get_session_id(request: HttpRequest) -> str:
    session_id = request.POST.get("session_id") or request.GET.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


def _rate_limit_ok(session_id: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    logs = _rate_logs.get(session_id, [])
    logs = [t for t in logs if t > window_start]
    if len(logs) >= RATE_LIMIT_MAX:
        return False
    logs.append(now)
    _rate_logs[session_id] = logs
    return True


def _get_session_data(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "messages": [],
            "profile": {
                "location": None,
                "weights": {},
                "min_scores": {},
                "mentioned_criteria": [],
                "follow_up_count": 0,
            },
            "turn": 0,
        }
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return SESSIONS[session_id]


def _extract_profile_from_tool_args(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Merge tool args into the profile dict."""
    profile: Dict[str, Any] = {}
    if tool_args.get("location_filter"):
        profile["location"] = tool_args["location_filter"]
    if tool_args.get("weights"):
        profile["weights"] = tool_args["weights"]
    if tool_args.get("min_scores"):
        profile["min_scores"] = tool_args["min_scores"]
    if tool_args.get("mentioned_criteria"):
        profile["mentioned_criteria"] = tool_args["mentioned_criteria"]
    return profile


def _validate_tool_args(tool_args: Dict[str, Any]) -> Optional[str]:
    """Validate and clamp tool arguments. Returns error message or None."""
    top_k = tool_args.get("top_k", 5)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(20, top_k))
    tool_args["top_k"] = top_k

    weights = tool_args.get("weights", {})
    if not isinstance(weights, dict):
        return "weights must be an object"
    valid_levels = {"low", "medium", "high", "must_have"}
    clean_weights: Dict[str, str] = {}
    for cid, level in weights.items():
        if cid not in CRITERION_IDS:
            return f"Unknown criterion: {cid}"
        if level not in valid_levels:
            return f"Invalid importance level for {cid}: {level}"
        clean_weights[cid] = level
    tool_args["weights"] = clean_weights

    min_scores = tool_args.get("min_scores", {})
    if not isinstance(min_scores, dict):
        return "min_scores must be an object"
    clean_min: Dict[str, float] = {}
    for cid, val in min_scores.items():
        if cid not in CRITERION_IDS:
            return f"Unknown criterion in min_scores: {cid}"
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return f"Invalid min value for {cid}: {val}"
        fval = max(0.0, min(100.0, fval))
        clean_min[cid] = fval
    tool_args["min_scores"] = clean_min

    mentioned = tool_args.get("mentioned_criteria", [])
    if not isinstance(mentioned, list):
        return "mentioned_criteria must be an array"
    tool_args["mentioned_criteria"] = [c for c in mentioned if c in CRITERION_IDS]

    return None


# ─── LLM client singleton ───────────────────────────────────────────────────
_llm_client: Optional[LLMClient] = None


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ─── Chat view ──────────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
def chat_view(request: HttpRequest) -> JsonResponse:
    """
    POST /api/chat/
    Body: {"message": "...", "session_id": "optional"}
    Returns: {"reply": "...", "zones": [...], "profile": {...}}
    """
    if request.method == "OPTIONS":
        return JsonResponse({})

    # Rate limit check
    session_id = _get_session_id(request)
    if not _rate_limit_ok(session_id):
        return JsonResponse(
            {"error": "Rate limit exceeded. Please slow down."},
            status=429,
        )

    # Parse body
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_message = body.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "message is required"}, status=400)

    client = _get_llm_client()
    session = _get_session_data(session_id)
    session["turn"] += 1

    # Append user message to conversation
    session["messages"].append({"role": "user", "content": user_message})

    # Run the tool-calling loop
    reply_text: Optional[str] = None
    zones_result: List[Dict[str, Any]] = []
    debug_info: Dict[str, Any] = {}
    tool_was_called = False

    max_loops = 5
    for _ in range(max_loops):
        llm_resp: LLMResponse = client.chat(
            system_prompt=build_system_prompt(),
            messages=session["messages"],
            tools=TOOL_DEFINITIONS,
        )

        if llm_resp.finish_reason == "error":
            reply_text = llm_resp.text or "Désolé, une erreur technique est survenue."
            break

        if llm_resp.finish_reason == "tool_call":
            tool_was_called = True
            tool_name = llm_resp.tool_name
            tool_args = dict(llm_resp.tool_args)

            # Validate tool arguments
            err = _validate_tool_args(tool_args)
            if err:
                # Feed the error back to the LLM
                session["messages"].append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": tool_name, "arguments": json.dumps(tool_args)}],
                })
                session["messages"].append({
                    "role": "tool",
                    "content": f"Erreur de validation : {err}",
                })
                continue

            # Execute search_zones
            try:
                ranked_zones, debug = search_zones(
                    location_filter=tool_args.get("location_filter"),
                    min_scores=tool_args.get("min_scores"),
                    weights=tool_args.get("weights"),
                    top_k=tool_args.get("top_k", 5),
                    mentioned_criteria=tool_args.get("mentioned_criteria"),
                )
            except Exception as exc:  # noqa: BLE001
                session["messages"].append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": tool_name, "arguments": json.dumps(tool_args)}],
                })
                session["messages"].append({
                    "role": "tool",
                    "content": f"Erreur lors de la recherche : {exc}",
                })
                continue

            zones_result = [rz.to_dict() for rz in ranked_zones]
            debug_info = debug

            # Update the session profile from tool args
            profile_update = _extract_profile_from_tool_args(tool_args)
            session["profile"].update(profile_update)
            if tool_args.get("location_filter"):
                session["profile"]["location"] = tool_args["location_filter"]
            session["profile"]["follow_up_count"] = 0  # reset after successful search

            # Feed tool result back to the LLM so it can write a text explanation
            session["messages"].append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": tool_name, "arguments": json.dumps(tool_args)}],
            })
            tool_result_summary = {
                "zones": zones_result,
                "debug": debug_info,
            }
            session["messages"].append({
                "role": "tool",
                "content": json.dumps(tool_result_summary, ensure_ascii=False),
            })
            continue

        # finish_reason == "text"
        reply_text = llm_resp.text
        break

    # If LLM asked a follow-up (text reply without tool call), record it
    if reply_text and not tool_was_called:
        session["profile"]["follow_up_count"] = session["profile"].get("follow_up_count", 0) + 1

    # Append assistant reply to conversation
    if reply_text:
        session["messages"].append({"role": "assistant", "content": reply_text})

    # Build profile response for the frontend
    profile_response = {
        "location": session["profile"].get("location"),
        "weights": session["profile"].get("weights", {}),
        "min_scores": session["profile"].get("min_scores", {}),
        "mentioned_criteria": session["profile"].get("mentioned_criteria", []),
        "follow_up_count": session["profile"].get("follow_up_count", 0),
    }

    return JsonResponse({
        "reply": reply_text or "",
        "zones": zones_result,
        "profile": profile_response,
        "debug": debug_info,
        "session_id": session_id,
    })