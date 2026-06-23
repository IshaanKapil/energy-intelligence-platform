"""
chat.py — AI Copilot for the Energy Intelligence Platform.

Uses Groq (llama-3.3-70b-versatile) to answer natural language questions
about energy data, grounding answers in SHAP drivers and platform context.

Usage:
    from copilot.chat import ask_copilot
    reply = ask_copilot("Why is load high today?")
"""

import json
from pathlib import Path
from typing import Optional

from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

MODEL = "llama-3.3-70b-versatile"


def _load_context() -> dict:
    """Build a context dict from available report files."""
    ctx = {}
    shap_path = REPORTS / "shap_top_drivers.json"
    if shap_path.exists():
        try:
            ctx["shap_drivers"] = json.loads(shap_path.read_text())
        except Exception:
            pass
    return ctx


def _build_system_prompt(context: Optional[dict] = None) -> str:
    if context is None:
        context = _load_context()

    parts = [
        "You are an AI energy analyst copilot for an Energy Intelligence Platform.",
        "You help operators understand electricity load patterns, solar generation,",
        "battery dispatch optimization, and anomaly detection results.",
        "",
        "Be concise and data-driven. When referencing numbers, be specific.",
        "If asked about things outside energy/power systems, politely redirect.",
        "",
    ]

    if context.get("shap_drivers"):
        parts.append("## Top Load Drivers (SHAP feature importance)")
        for d in context["shap_drivers"]:
            parts.append(f"  - {d['feature']}: mean |SHAP| = {d['mean_abs_shap']:.3f} MW")
        parts.append("")

    parts += [
        "## Platform Context",
        "- Load forecasting: LightGBM with Conformalized Quantile Regression (80% intervals)",
        "- Solar forecasting: LightGBM trained on Open-Meteo weather data",
        "- Battery optimization: OR-Tools GLOP Linear Programming",
        "- Anomaly detection: Isolation Forest (contamination=2%)",
        "- Pricing proxy: $80/MWh peak (8am-8pm weekdays), $40/MWh off-peak",
        "- Data source: PJM AEP historical load + Open-Meteo weather",
    ]

    return "\n".join(parts)


def ask_copilot(
    question: str,
    context: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> str:
    """
    Ask the AI copilot a question about energy data.

    Args:
        question: Natural language question from the user
        context: Optional dict with additional context
        conversation_history: Optional list of prior {role, content} messages

    Returns:
        AI response string
    """
    client = Groq()

    system = _build_system_prompt(context)

    messages = [{"role": "system", "content": system}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content


def stream_copilot(
    question: str,
    context: Optional[dict] = None,
    conversation_history: Optional[list] = None,
):
    """
    Stream the AI copilot response token-by-token.

    Yields text chunks as they arrive.
    """
    client = Groq()

    system = _build_system_prompt(context)

    messages = [{"role": "system", "content": system}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
