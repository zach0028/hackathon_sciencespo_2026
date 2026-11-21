"""Tests for Gemini 3.8 Live + Gemini 4 Neon brain integration and provider switcher."""

from __future__ import annotations

import asyncio

from hackathon_osmo.ai.gemini_brain import (
    GEMINI_BRAIN_MODELS,
    GEMINI_LIVE_MODEL,
    GEMINI_LIVE_VOICES,
    GeminiLiveBrain,
    extract_gemini_text_and_calls,
    openai_tools_to_gemini,
)
from hackathon_osmo.ai.live_brain import ROBOT_TOOLS
from hackathon_osmo.app.runtime import StudioRuntime


def test_openai_tools_to_gemini_preserves_all_tools_and_strips_strict() -> None:
    live_decls = openai_tools_to_gemini(ROBOT_TOOLS, non_blocking=True)
    rest_decls = openai_tools_to_gemini(ROBOT_TOOLS, non_blocking=False)

    assert len(live_decls) == len(ROBOT_TOOLS) == 7
    names = {d["name"] for d in live_decls}
    assert {
        "look_at_user",
        "look_at_desk",
        "move_arm",
        "express",
        "set_tracking",
        "remember",
        "show_on_screen",
    } == names

    for d in live_decls:
        assert d["behavior"] == "NON_BLOCKING"
        assert "strict" not in d
        assert d["parameters"]["type"] == "OBJECT"
        assert "additionalProperties" not in d["parameters"]

    for d in rest_decls:
        assert "behavior" not in d
        assert d["parameters"]["type"] == "OBJECT"


def test_extract_gemini_text_and_calls_skips_thoughts() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "internal chain of thought", "thought": True},
                        {
                            "functionCall": {
                                "name": "show_on_screen",
                                "args": {
                                    "face_user": True,
                                    "mode": "definition",
                                    "title": "QMAX",
                                    "items": ["Heat flux peak"],
                                    "note": "",
                                    "spoken_summary": "Here is Qmax on my screen.",
                                },
                            }
                        },
                        {"text": "Here is Qmax on my screen."},
                    ],
                }
            }
        ]
    }
    text, calls, raw = extract_gemini_text_and_calls(payload)
    assert text == "Here is Qmax on my screen."
    assert len(calls) == 1
    assert calls[0]["name"] == "show_on_screen"
    assert calls[0]["arguments"]["face_user"] is True
    assert len(raw) == 3


def test_gemini_brain_language_and_instructions() -> None:
    brain = GeminiLiveBrain()
    assert brain.live_model == GEMINI_LIVE_MODEL
    assert "ENGLISH ONLY" in brain.live_instructions()
    asyncio.run(brain.set_language("fr"))
    assert "FRANÇAIS UNIQUEMENT" in brain.live_instructions()
    assert "FRANÇAIS" in brain.brain_instructions()


def test_runtime_provider_switch_syncs_language_and_snapshot() -> None:
    rt = StudioRuntime()
    asyncio.run(rt.set_ai_provider("gemini"))
    asyncio.run(rt.openai_brain.set_language("fr"))
    asyncio.run(rt.gemini_brain.set_language("fr"))

    snap_gem = rt.snapshot()["voice"]
    assert snap_gem["provider"] == "gemini"
    assert snap_gem["models"] == GEMINI_BRAIN_MODELS
    assert snap_gem["voices"] == GEMINI_LIVE_VOICES
    assert snap_gem["language"] == "fr"

    asyncio.run(rt.set_ai_provider("openai"))
    snap_oai = rt.snapshot()["voice"]
    assert snap_oai["provider"] == "openai"
    assert snap_oai["language"] == "fr"
    assert any(m["id"] == "gpt-6.2-sol" for m in snap_oai["models"])
