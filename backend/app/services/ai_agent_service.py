"""
ai_agent_service.py
===================
LangChain + ChatGroq (LLaMA-3) powered AI agent for:
1. Intelligent dispatch recommendations with narrative reasoning
2. Risk escalation analysis across zones
3. Persistent zone threat briefing
4. Intercept feasibility analysis
"""

import os
import json
import math
from typing import Any
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PROTECTED_AREAS_NAMES = [
    "Mesoamerican Reef",
    "Bay Islands MPA",
    "Sian Ka'an Reserve",
    "Great Barrier Reef",
    "Galápagos Marine Reserve",
    "Papahānaumokuākea",
    "Mediterranean MPA Network",
    "Chagos Marine Reserve",
]


def _build_zone_summary(clusters: list, dispatches: list = None) -> str:
    """Build a concise text summary of debris zones for the LLM."""
    lines = []
    for i, c in enumerate(clusters[:10]):  # cap at 10 zones
        mpa = ""
        if dispatches:
            d = next((d for d in dispatches if d.get("zone_id") == c.get("id", i)), None)
            if d and d.get("nearest_mpa"):
                mpa = f" | near {d['nearest_mpa']} ({d.get('nearest_mpa_km', '?')}km)"
        lines.append(
            f"Zone #{c.get('id', i) + 1}: priority={c.get('priority','?')}, "
            f"density={c.get('density', 0)} pts, radius={c.get('radius_m', 0):.0f}m, "
            f"persistence={c.get('persistence', False)}, "
            f"recency={c.get('recency_hours', 0):.1f}h ago, "
            f"score={c.get('priority_score', 0):.2f}{mpa}"
        )
    return "\n".join(lines)


def _build_dispatch_summary(dispatches: list) -> str:
    lines = []
    for d in dispatches[:10]:
        threats_text = "; ".join(t.get("message", "") for t in d.get("threats", []))
        lines.append(
            f"Zone #{d.get('zone_id', 0) + 1} -> vessel={d.get('assigned_vessel')}, "
            f"urgency={d.get('urgency')}, crew={d.get('vessel_crew')}, "
            f"est_cleanup={d.get('estimated_cleanup_hours')}h, "
            f"mpa_risk={d.get('mpa_risk', 'NONE')}"
            + (f", threats=[{threats_text}]" if threats_text else "")
        )
    return "\n".join(lines)


def analyze_dispatch_with_ai(clusters: list, dispatches: list, summary: dict) -> dict:
    """
    Use ChatGroq LLaMA3 to generate an intelligent operational briefing
    for the dispatch plan, including narrative reasoning, prioritization
    logic, and recommended escalations.
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured", "analysis": None}

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=1500,
        )

        zone_text = _build_zone_summary(clusters, dispatches)
        dispatch_text = _build_dispatch_summary(dispatches)

        system_prompt = """You are NAUTILUS, an AI-powered Coast Guard Operations Analyst for a marine debris cleanup intelligence system.

Your mission: analyze satellite-detected debris zones and provide a concise, actionable operational briefing to coast guard commanders.

Rules:
- Be direct, military-style, operational
- Highlight zones that threaten Marine Protected Areas first
- Identify patterns (persistent zones, drift corridors)
- Give specific numbered recommendations
- Keep response under 400 words
- Use the structured JSON format provided"""

        human_prompt = f"""SITUATION REPORT:
Total detections: {summary.get('total_points', 0)}
Active cleanup zones: {summary.get('total_clusters', 0)}
Immediate action required: {summary.get('high', 0)} zones
Deploy scheduled: {summary.get('medium', 0)} zones

ZONE INTELLIGENCE:
{zone_text}

DISPATCH ASSIGNMENTS:
{dispatch_text}

Generate an operational briefing in this JSON format:
{{
  "threat_level": "CRITICAL|HIGH|MEDIUM|LOW",
  "sitrep": "2-3 sentence situation summary",
  "priority_zones": [list of zone IDs requiring immediate attention, e.g. [1,2]],
  "mpa_risk_zones": [zone IDs near protected areas],
  "recommendations": ["numbered action items, 3-5 items"],
  "persistent_threat_assessment": "one sentence on chronic accumulation zones",
  "estimated_ops_window": "estimated hours/days for full cleanup"
}}"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        raw = response.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Find JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        parsed = json.loads(raw)
        return {"status": "success", "analysis": parsed, "model": "llama-3.3-70b-versatile"}

    except json.JSONDecodeError as e:
        return {"status": "partial", "analysis": {"sitrep": response.content[:500]}, "error": f"JSON parse error: {e}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "analysis": None}


def analyze_persistent_zones_with_ai(zones: list) -> dict:
    """Use ChatGroq to generate a persistent zone threat briefing."""
    if not GROQ_API_KEY or not zones:
        return {"error": "No data or API key missing", "analysis": None}

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=800,
        )

        zone_lines = []
        for i, z in enumerate(zones[:8]):
            t = z.get("threat", {})
            zone_lines.append(
                f"Zone #{i+1}: lat={z['lat']:.3f}°, lon={z['lon']:.3f}°, "
                f"intervals_active={z['intervals_active']}, "
                f"total_detections={z['total_detections']}, "
                f"threat={t.get('level','?')}"
                + (f", near {t.get('nearest_mpa')} ({t.get('nearest_mpa_km')}km)" if t.get('nearest_mpa') else "")
            )

        system = "You are a marine environmental analyst. Analyze persistent debris accumulation patterns and assess ecosystem risk."
        human = f"""Persistent debris zones detected over 7-day analysis:
{chr(10).join(zone_lines)}

Respond in JSON:
{{
  "ecosystem_risk": "CRITICAL|HIGH|MEDIUM|LOW",
  "pattern_analysis": "one sentence on spatial/temporal patterns",
  "highest_risk_zone": zone number (integer),
  "recommended_priority": "which zone to address first and why (1 sentence)",
  "long_term_action": "one sentence on long-term monitoring/cleanup recommendation"
}}"""

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)
        return {"status": "success", "analysis": parsed}

    except Exception as e:
        return {"status": "error", "error": str(e), "analysis": None}


def analyze_intercept_with_ai(
    debris_origin: dict,
    vessel_origin: dict,
    intercept_point: dict,
    trajectory: list,
    vessel_speed: float,
) -> dict:
    """Generate AI narrative for an intercept plan."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured", "analysis": None}

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=500,
        )

        # Calculate drift speed
        if len(trajectory) >= 2:
            p0, p1 = trajectory[0], trajectory[1]
            drift_km = math.sqrt((p1["lat"] - p0["lat"]) ** 2 + (p1["lon"] - p0["lon"]) ** 2) * 111
        else:
            drift_km = 0.1

        human = f"""INTERCEPT SCENARIO:
Debris origin: {debris_origin.get('lat', 0):.4f}°, {debris_origin.get('lon', 0):.4f}°
Vessel position: {vessel_origin.get('lat', 0):.4f}°, {vessel_origin.get('lon', 0):.4f}°
Vessel speed: {vessel_speed} knots
Intercept point: {intercept_point.get('intercept_lat', 0):.4f}°, {intercept_point.get('intercept_lon', 0):.4f}°
Intercept hour: T+{intercept_point.get('intercept_hour', 0)}h
Travel distance: {intercept_point.get('vessel_travel_km', 0)} km
Travel time: {intercept_point.get('vessel_travel_hours', 0)}h
Debris drift: ~{drift_km:.2f} km/h

In JSON:
{{
  "feasibility": "OPTIMAL|FEASIBLE|MARGINAL|INFEASIBLE",
  "tactical_assessment": "1-2 sentence assessment",
  "risk_window": "how long before debris drifts out of intercept range",
  "recommended_action": "specific action for the vessel commander"
}}"""

        response = llm.invoke([
            SystemMessage(content="You are a Coast Guard tactical operations officer. Be concise and actionable."),
            HumanMessage(content=human),
        ])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)
        return {"status": "success", "analysis": parsed}
    except Exception as e:
        return {"status": "error", "error": str(e), "analysis": None}
