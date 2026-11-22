"""Unit tests for OpenAI GPT-Live 1.1 + GPT-6.2 Astra delegated brain and native tool loop."""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np

from hackathon_osmo.ai.live_brain import (
    ROBOT_TOOLS,
    OpenAILiveBrain,
    extract_live_session_info,
    extract_responses_text_and_calls,
    infer_micro_expression,
    prepare_vision_jpeg,
)


def test_robot_tools_compact_five_tool_hierarchy() -> None:
    names = [t["name"] for t in ROBOT_TOOLS]
    assert names == [
        "look_at_user",
        "look_at_desk",
        "move_arm",
        "express",
        "set_tracking",
        "remember",
        "show_on_screen",
        "set_situation",
    ]
    for t in ROBOT_TOOLS:
        assert t["strict"] is True
        assert t["parameters"]["additionalProperties"] is False

    desk_tool = next(t for t in ROBOT_TOOLS if t["name"] == "look_at_desk")
    assert "read_now" in desk_tool["parameters"]["properties"]
    assert "read_now" in desk_tool["parameters"]["required"]

    screen_tool = next(t for t in ROBOT_TOOLS if t["name"] == "show_on_screen")
    props = screen_tool["parameters"]["properties"]
    # `face_user` must be the FIRST schema property so SSE streams it within ~50ms for parallel arm return!
    assert list(props.keys())[0] == "face_user"
    assert props["mode"]["enum"] == [
        "definition",
        "concept",
        "exercise",
        "exam",
        "checklist",
        "feedback",
        "clear",
    ]
    assert "spoken_summary" in props
    assert set(screen_tool["parameters"]["required"]) == {
        "face_user",
        "mode",
        "title",
        "items",
        "note",
        "spoken_summary",
    }


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
    assert max(hd_img.shape[:2]) == 1600


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


def test_stream_deltas_drive_zero_latency_micro_states() -> None:
    lights: list[str] = []
    brain = OpenAILiveBrain(on_light=lights.append)

    brain._handle_stream_event("response.in_progress", {}, "thinking")
    assert brain.micro_phase == "thinking"

    brain._handle_stream_event("response.reasoning_summary_text.delta", {"delta": "step 1"}, "thinking")
    assert brain.micro_phase == "deep_thinking"
    p1 = brain.micro_pulse_seq
    assert p1 >= 1

    brain._handle_stream_event(
        "response.output_item.added",
        {"item": {"type": "function_call", "name": "look_at_desk"}},
        "thinking",
    )
    assert brain.micro_phase == "tool_intent"
    assert brain.micro_tool_intent == "look_at_desk"
    assert brain.micro_pulse_seq == p1 + 1
    assert lights == ["curious"]

    brain._handle_stream_event("response.output_text.delta", {"delta": "Je vois"}, "reading_sheet")
    assert brain.micro_phase == "formulating"
    assert brain.micro_pulse_seq == p1 + 2


def test_sideband_detects_attentive_listening_and_barge_in() -> None:
    brain = OpenAILiveBrain()

    async def _run() -> None:
        await brain._handle_sideband_event({"type": "session.input_transcript.delta", "delta": "Salut "})
        snap = brain.micro_snapshot()
        assert snap["phase"] == "user_speaking"
        assert snap["pulse_seq"] >= 1

        # Osmo starts speaking -> user interrupts mid-sentence -> barge_in!
        await brain._handle_sideband_event({"type": "session.output_transcript.delta", "delta": "Bonjour "})
        assert brain.speaking is True
        await brain._handle_sideband_event({"type": "input_audio_buffer.speech_started"})
        snap2 = brain.micro_snapshot()
        assert brain.speaking is False
        assert snap2["phase"] == "barge_in"
        assert snap2["barge_in_age_ms"] is not None and snap2["barge_in_age_ms"] < 1000.0

    asyncio.run(_run())


def test_sse_stream_triggers_speculative_hardware_prewarm() -> None:
    prewarms: list[str] = []
    brain = OpenAILiveBrain(on_prewarm=prewarms.append)

    # 1. look_at_desk immediately pre-warms desk pivot
    brain._handle_stream_event(
        "response.output_item.added",
        {"item": {"type": "function_call", "name": "look_at_desk"}},
        "thinking",
    )
    assert prewarms == ["desk"]

    # 2. show_on_screen with face_user=true pre-warms face return on the first JSON delta!
    brain._stream_prewarm_fired = ""
    brain._handle_stream_event(
        "response.output_item.added",
        {"item": {"type": "function_call", "name": "show_on_screen"}},
        "reading_sheet",
    )
    brain._handle_stream_event(
        "response.function_call_arguments.delta",
        {"delta": '{"face_user": true, "mode": "exercise"'},
        "reading_sheet",
    )
    assert prewarms == ["desk", "face"]

    # 3. show_on_screen with face_user=false must NOT pre-warm face return
    brain._stream_prewarm_fired = ""
    brain._handle_stream_event(
        "response.output_item.added",
        {"item": {"type": "function_call", "name": "show_on_screen"}},
        "reading_sheet",
    )
    brain._handle_stream_event(
        "response.function_call_arguments.delta",
        {"delta": '{"face_user": false, "mode": "concept"'},
        "reading_sheet",
    )
    assert prewarms == ["desk", "face"]


def test_infer_micro_expression_and_realtime_sideband_auto_express() -> None:
    assert infer_micro_expression("Yes, exactly! Great job.") == ("nod_yes", "yes")
    assert infer_micro_expression("Hello everyone! Nice to meet you.") == ("nod_yes", "yes")
    assert infer_micro_expression("Not quite, watch the minus sign.") == ("shake_no", "no")
    assert infer_micro_expression("What do you think we should try next?", final=True) == ("curious_tilt", "curious")
    assert infer_micro_expression("Oui, exactement ! Bravo.") == ("nod_yes", "yes")
    assert infer_micro_expression("Pas tout à fait, attention au signe.") == ("shake_no", "no")
    assert infer_micro_expression("Intéressant, on essaie ensemble ?", final=True) == ("curious_tilt", "curious")
    assert infer_micro_expression("Heat flows from warm skin to fabric.", final=True) is None

    # Language switch toggles between full English and full French instructions cleanly
    b_lang = OpenAILiveBrain()
    assert b_lang.language == "en"
    assert "ENGLISH ONLY" in b_lang.live_instructions()
    assert "ENGLISH ONLY" in b_lang.brain_instructions()
    asyncio.run(b_lang.set_language("fr"))
    assert b_lang.language == "fr"
    assert "FRANÇAIS UNIQUEMENT" in b_lang.live_instructions()
    assert "FRANÇAIS CLAIR" in b_lang.brain_instructions()

    fired: list[tuple[str, str]] = []
    brain = OpenAILiveBrain(on_auto_express=lambda g, l: fired.append((g, l)))

    async def _run() -> None:
        await brain._handle_sideband_event({"type": "session.output_transcript.delta", "delta": "Hi, Jules! "})
        await brain._handle_sideband_event({"type": "session.output_transcript.delta", "delta": "Nice to meet you."})
        await brain._handle_sideband_event({"type": "session.output_transcript.done"})
        assert fired == [("nod_yes", "yes")]

        # Next utterance: question ending with '?' fires curious_tilt on done!
        await brain._handle_sideband_event({"type": "session.output_transcript.delta", "delta": "Shall we try Q2?"})
        await brain._handle_sideband_event({"type": "session.output_transcript.done"})
        assert fired == [("nod_yes", "yes"), ("curious_tilt", "curious")]

    asyncio.run(_run())




