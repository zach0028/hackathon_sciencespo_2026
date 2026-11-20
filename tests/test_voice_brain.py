"""Unit tests for the VoiceBrain JSON parser and safety guards."""

from __future__ import annotations

from hackathon_osmo.ai.voice_brain import _ensure_wav_16k, _parse_brain_json


def test_parse_clean_json() -> None:
    raw = '{"reply": "Oui bien sûr, je suis prêt !", "gesture": "nod_yes", "led": "yes"}'
    reply, gesture, led = _parse_brain_json(raw)
    assert reply == "Oui bien sûr, je suis prêt !"
    assert gesture == "nod_yes"
    assert led == "yes"


def test_parse_markdown_fenced_json() -> None:
    raw = '```json\n{"reply": "Non, ce n\'est pas exact.", "gesture": "shake_no", "led": "no"}\n```'
    reply, gesture, led = _parse_brain_json(raw)
    assert reply == "Non, ce n'est pas exact."
    assert gesture == "shake_no"
    assert led == "no"


def test_parse_filters_unknown_gesture_and_led() -> None:
    raw = '{"reply": "Salut !", "gesture": "backflip", "led": "rainbow"}'
    reply, gesture, led = _parse_brain_json(raw)
    assert reply == "Salut !"
    assert gesture is None
    assert led is None


def test_parse_truncated_json_never_leaks_braces_to_tts() -> None:
    raw = '{"reply": "Je regarde ta feuille tout de suite", "gesture": "look_desk'
    reply, gesture, led = _parse_brain_json(raw)
    assert reply == "Je regarde ta feuille tout de suite"
    assert "{" not in reply and "}" not in reply
    assert gesture is None
    assert led is None


def test_ensure_wav_16k_passes_through_valid_wav_header() -> None:
    fake_wav = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 40
    assert _ensure_wav_16k(fake_wav) == fake_wav
