"""Unit tests for OpenAI GPT-Live 1.1 + GPT-6.2 Astra delegated brain and native tool loop."""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np

from hackathon_osmo.ai.live_brain import (
    OpenAILiveBrain,
    extract_live_session_info,
    extract_responses_text_and_calls,
    prepare_vision_jpeg,
)


def test_extract_live_session_info_supports_nested_and_flat_shapes() -> None:
    nested = {"session": {"id": "sess_123"}, "transport": {"type": "webrtc", "sdp": "v=0\r\nm=audio"}}
    assert extract_live_session_info(nested) == ("sess_123", "v=0\r\nm=audio")

    flat = {"id": "sess_456", "sdp": "v=0\r\nflat"}
    assert extract_live_session_info(flat) == ("sess_456", "v=0\r\nflat")


def test_extract_responses_text_and_function_calls() -> None:
    payload = {
        "id": "resp_1",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "trigger_gesture",
                "arguments": json.dumps({"gesture": "nod_yes"}),
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "set_hilight",
                "arguments": json.dumps({"effect": "yes"}),
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Oui, tout à fait !"}],
            },
        ],
    }
    text, calls = extract_responses_text_and_calls(payload)
    assert text == "Oui, tout à fait !"
    assert len(calls) == 2
    assert calls[0]["name"] == "trigger_gesture" and calls[0]["arguments"] == {"gesture": "nod_yes"}
    assert calls[1]["name"] == "set_hilight" and calls[1]["arguments"] == {"effect": "yes"}


def test_prepare_vision_jpeg_downscales_standard_and_keeps_hd() -> None:
    img = np.zeros((1512, 2688, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    raw = enc.tobytes()

    low_bytes = prepare_vision_jpeg(raw, high_res=False)
    hd_bytes = prepare_vision_jpeg(raw, high_res=True)
    assert low_bytes is not None and hd_bytes is not None

    low_img = cv2.imdecode(np.frombuffer(low_bytes, np.uint8), cv2.IMREAD_COLOR)
    hd_img = cv2.imdecode(np.frombuffer(hd_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert max(low_img.shape[:2]) == 640
    assert max(hd_img.shape[:2]) == 2048


def test_reasoning_param_enforces_model_floor() -> None:
    b = OpenAILiveBrain(brain_model="gpt-6.2-astra", reasoning_effort="none")
    assert b._build_reasoning_param() == {"effort": "low"}

    b_luna = OpenAILiveBrain(brain_model="gpt-6-luna", reasoning_effort="low")
    assert b_luna._build_reasoning_param() == {"effort": "none"}


def test_sideband_event_accumulates_transcripts_and_deduplicates() -> None:
    brain = OpenAILiveBrain()

    async def _run() -> None:
        await brain._handle_sideband_event({"type": "session.input_transcript.delta", "delta": "Est-ce que "})
        await brain._handle_sideband_event({"type": "session.input_transcript.delta", "delta": "2+2 font 5 ?"})
        await brain._handle_sideband_event({"type": "session.input_transcript.done"})
        assert brain._latest_user_utterance() == "Est-ce que 2+2 font 5 ?"
        assert len(brain.transcript) == 1
        assert brain.transcript[0]["user"] == "Est-ce que 2+2 font 5 ?"

        await brain._handle_sideband_event({"type": "session.output_transcript.done", "transcript": "Non, ça fait 4 !"})
        assert brain.transcript[0]["reply"] == "Non, ça fait 4 !"

    asyncio.run(_run())


def test_consume_user_utterance_clears_buffer_between_delegations() -> None:
    brain = OpenAILiveBrain()
    brain._user_transcript_buf = "Tourne la caméra à gauche"
    first = brain._consume_user_utterance()
    assert first == "Tourne la caméra à gauche"
    assert brain._user_transcript_buf == ""

    brain._user_transcript_buf = "Maintenant regarde en haut"
    second = brain._consume_user_utterance()
    assert second == "Maintenant regarde en haut"

