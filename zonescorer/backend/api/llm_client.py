"""
ZoneScore — LLM client for the chat agent.

Supports two backends:
  1. Mock (default): deterministic keyword-based extraction, works without any API key.
     Useful for demos, hackathons, and offline testing.
  2. Google AI Studio (Gemini): set GOOGLE_API_KEY to use a real Gemini model.

The tool-calling loop lives in chat_views.py; this module only handles single-turn
LLM calls (system prompt + messages + tool definitions → text or tool call).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zonescorer.env import config, config_bool


@dataclass
class LLMResponse:
    text: Optional[str] = None            # plain text reply (no tool call)
    tool_name: Optional[str] = None       # name of the tool the LLM wants to call
    tool_args: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "text"           # "text" | "tool_call" | "error"
    raw: str = ""                         # raw response for debugging
    raw_part: Optional[Dict[str, Any]] = None  # original Gemini part (preserves thoughtSignature)


class LLMClient:
    """Thin wrapper over either a mock LLM or Google's Gemini API."""

    FALLBACK_MODELS = (
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
    )

    def __init__(self):
        self.use_mock = config_bool("ZONESCORER_LLM_MOCK", default=True)
        self.google_api_key = config("GOOGLE_API_KEY", default="").strip()
        self.model = config("ZONESCORER_LLM_MODEL", default="gemini-3-flash-preview").strip()
        self.temperature = float(config("ZONESCORER_LLM_TEMPERATURE", default="0.3"))
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    # ─── Public API ─────────────────────────────────────────────────────────
    def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """Send a chat completion request; return text or a tool call."""
        if self.use_mock or not self.google_api_key:
            return self._mock_chat(system_prompt, messages, tools)
        return self._google_chat(system_prompt, messages, tools)

    # ─── Mock backend (deterministic, no API key needed) ────────────────────
    def _mock_chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """
        Deterministic keyword-based extraction for demo / offline use.

        Parses the last user message for:
          - location names (via the location resolver alias list)
          - criterion keywords (from criteria_config synonyms)
          - importance level cues (high / moyenne / importante / impératif etc.)

        Returns a text reply if the profile is incomplete, or a search_zones tool call.
        """
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        # ── If the previous turn was a tool result, write a text explanation ──
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "tool":
            try:
                tool_result = json.loads(last_msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                tool_result = {}
            zones = tool_result.get("zones", [])
            debug = tool_result.get("debug", {})
            if not zones:
                return LLMResponse(
                    text="Je n'ai trouvé aucune zone correspondant à vos critères. Voulez-vous élargir la recherche ou modifier vos préférences ?",
                    finish_reason="text",
                    raw="[mock] no zones from tool result",
                )
            loc = debug.get("location", {}).get("query", "cette région") if isinstance(debug, dict) else "cette région"
            top = zones[0]
            zname = top["zone"]["name"]
            score = top["match_score"]
            strongest = top.get("strongest")
            weakest = top.get("weakest")
            reasons = top.get("reasons", [])
            reason_text = " ".join(reasons) if reasons else ""
            explanation = (
                f"Voici les meilleures zones pour {loc}. "
                f"{zname} arrive en tête avec un score de compatibilité de {score}/100. {reason_text}"
            )
            return LLMResponse(
                text=explanation,
                finish_reason="text",
                raw=f"[mock] explanation for {len(zones)} zones",
            )

        # Lazy import to avoid circular imports.
        from .criteria_config import CRITERIA_CONFIG, CRITERION_IDS
        from .zone_data import ZONES, ZONE_BY_ID, _LOCATION_ALIASES
        from .location_resolver import resolve_location
        import unicodedata

        def norm(s: str) -> str:
            s = s.lower().strip()
            return "".join(
                ch for ch in unicodedata.normalize("NFD", s)
                if unicodedata.category(ch) != "Mn"
            )

        text_n = norm(user_text)

        # ── Detect location ────────────────────────────────────────────────
        location: Optional[str] = None
        loc_candidates = resolve_location(user_text, limit=5, min_score=0.5)
        if loc_candidates:
            exact = [c for c in loc_candidates if c["match_type"] == "exact"]
            location = (exact or loc_candidates)[0]["name"]

        # ── Detect criteria + importance levels ────────────────────────────
        level_cues: Dict[str, str] = {
            # French
            "très": "high", "fort": "high", "important": "high", "haut": "high",
            "impératif": "must_have", "obligatoire": "must_have", "indispensable": "must_have",
            "moyen": "medium", "moyenne": "medium", "assez": "medium", "modéré": "medium",
            "peu": "low", "faible": "low", "secondaire": "low", "priorité basse": "low",
            # Arabic (transliterated & script)
            "مهم": "high", "أولوية": "high", "أساسي": "high", "ضروري": "must_have",
            "مفضل": "medium", "مقبول": "medium", "أقل": "low", "ثانوي": "low",
            # English
            "very": "high", "crucial": "must_have", "mandatory": "must_have", "required": "must_have",
            "medium": "medium", "moderate": "medium", "little": "low", "minor": "low",
        }

        detected_levels: Dict[str, str] = {}
        detected_criteria: List[str] = []

        for cid, crit in CRITERIA_CONFIG.items():
            matched = False
            for syn_list in (crit.synonyms_fr, crit.synonyms_ar, crit.synonyms_en):
                for syn in syn_list:
                    if norm(syn) in text_n:
                        detected_criteria.append(cid)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                continue

            level = "medium"
            for cue, lvl in level_cues.items():
                if norm(cue) in text_n:
                    rank = {"must_have": 4, "high": 3, "medium": 2, "low": 1}
                    if rank[lvl] > rank.get(level, 0):
                        level = lvl
            detected_levels[cid] = level

        # ── Decide: ask follow-up or call search_zones? ────────────────────
        has_location = location is not None
        has_priorities = len(detected_criteria) > 0

        if not has_location or not has_priorities:
            missing = []
            if not has_location:
                missing.append("emplacement / gouvernorat / délégation (ex: La Marsa, Ariana, Tunis)")
            if not has_priorities:
                missing.append("priorités (calme, vert, accès, transport, budget...)")
            question = "Pour affiner votre recherche, pourriez-vous préciser " + " et ".join(missing) + " ?"
            return LLMResponse(
                text=question, finish_reason="text",
                raw=f"[mock] follow-up needed: location={has_location}, priorities={has_priorities}",
            )

        # ── Build search_zones arguments ───────────────────────────────────
        weights = {cid: detected_levels[cid] for cid in detected_criteria}
        min_scores: Dict[str, float] = {}
        for cid, level in detected_levels.items():
            if level == "must_have":
                min_scores[cid] = 60.0

        tool_args: Dict[str, Any] = {
            "location_filter": location,
            "weights": weights,
            "min_scores": min_scores,
            "top_k": 5,
            "mentioned_criteria": detected_criteria,
        }

        return LLMResponse(
            tool_name="search_zones",
            tool_args=tool_args,
            finish_reason="tool_call",
            raw=f"[mock] search_zones {tool_args}",
        )

    # ─── Google AI Studio (Gemini) backend ──────────────────────────────────
    @staticmethod
    def _sanitize_gemini_schema(schema: Any) -> Any:
        """Strip JSON-Schema keywords unsupported by Gemini functionDeclarations."""
        if isinstance(schema, dict):
            unsupported = {"additionalProperties", "default", "minimum", "maximum", "$schema"}
            return {
                k: LLMClient._sanitize_gemini_schema(v)
                for k, v in schema.items()
                if k not in unsupported
            }
        if isinstance(schema, list):
            return [LLMClient._sanitize_gemini_schema(item) for item in schema]
        return schema

    def _google_chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """Call Google's Generative AI API with proper Gemini function calling."""
        function_declarations = []
        for t in tools:
            fn = t.get("function", {})
            params = self._sanitize_gemini_schema(
                fn.get("parameters", {"type": "object", "properties": {}})
            )
            function_declarations.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": params,
            })

        # Build Gemini contents from conversation history.
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "tool":
                tool_name = m.get("tool_name") or "search_zones"
                raw_content = m.get("content", "{}")
                try:
                    parsed_result = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                except (json.JSONDecodeError, TypeError):
                    parsed_result = {"output": str(raw_content)}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": {"result": parsed_result},
                        }
                    }],
                })
                continue

            if role == "assistant" and m.get("tool_calls"):
                parts = []
                if m.get("raw_part"):
                    parts.append(m["raw_part"])
                else:
                    for tc in m["tool_calls"]:
                        raw_args = tc.get("arguments", "{}")
                        try:
                            args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except (json.JSONDecodeError, TypeError):
                            args_obj = {}
                        parts.append({
                            "functionCall": {
                                "name": tc.get("name", "search_zones"),
                                "args": args_obj,
                            }
                        })
                contents.append({"role": "model", "parts": parts})
                continue

            google_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": google_role,
                "parts": [{"text": m.get("content", "")}],
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048,
            },
        }
        if function_declarations:
            payload["tools"] = [{"functionDeclarations": function_declarations}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        data = json.dumps(payload).encode("utf-8")
        candidate_models = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        body = None
        last_err = ""

        for model_name in candidate_models:
            url = (
                f"{self.base_url}/models/{model_name}:generateContent"
                f"?key={self.google_api_key}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                last_err = f"HTTP {exc.code} ({model_name}): {err_body[:300]}"
                if exc.code in (404, 429, 500, 503):
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__} ({model_name}): {exc}"
                continue

        if body is None:
            return LLMResponse(
                text="Désolé, je rencontre un problème technique avec Gemini. Réessayez plus tard.",
                finish_reason="error",
                raw=last_err,
            )

        candidates = body.get("candidates", [])
        if not candidates:
            return LLMResponse(
                text="Je n'ai pas pu générer de réponse. Pouvez-vous reformuler ?",
                finish_reason="error",
                raw=json.dumps(body),
            )

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Check for a function call (preserve raw part for Gemini 3 thoughtSignature).
        for part in parts:
            fc = part.get("functionCall")
            if fc:
                return LLMResponse(
                    tool_name=fc.get("name", ""),
                    tool_args=dict(fc.get("args", {})),
                    finish_reason="tool_call",
                    raw=json.dumps(body),
                    raw_part=part,
                )

        # Otherwise return text.
        text_parts = [p.get("text", "") for p in parts if p.get("text")]
        return LLMResponse(
            text="".join(text_parts),
            finish_reason="text",
            raw=json.dumps(body),
        )