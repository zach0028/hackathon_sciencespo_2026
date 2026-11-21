"""OpenAI Full-Duplex Voice + Delegated Frontier Vision Brain.

Architecture (Topology A):
- Voice frontend : gpt-live-1.1 over WebRTC (UDP/Opus, native full-duplex barge-in)
- Control bridge : Python Sideband WebSocket (/v1/live/sessions/{id}/attach)
- Reasoning brain: gpt-6.2-astra (reasoning.effort="low") via OpenAI /v1/responses
  with persistent HTTP keep-alive, Pixel 11 Pro live camera frames, and native
  function tools driving the DJI Osmo Mobile 3 & Pixel HiLight 8-LED ring.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import cv2
import httpx
import numpy as np
import websockets
from dotenv import load_dotenv

from hackathon_osmo.ai.memory import OsmoMemoryStore
from hackathon_osmo.paths import ENV_FILE
from hackathon_osmo.telemetry.recorder import SessionRecorder

load_dotenv(ENV_FILE)

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_WS_BASE = "wss://api.openai.com/v1"

LIVE_MODEL = "gpt-live-1.1"
DEFAULT_BRAIN_MODEL = "gpt-6.2-sol"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_LIVE_VOICE = "cedar"
DEFAULT_LANGUAGE = "en"  # "en" (Full English) | "fr" (Full Français)

LIVE_VOICES = [
    {"id": "cedar", "label": "Cedar (Male · Recommended)"},
    {"id": "ash", "label": "Ash (Male · Calm)"},
    {"id": "verse", "label": "Verse (Male · Expressive)"},
    {"id": "marin", "label": "Marin (Female)"},
    {"id": "coral", "label": "Coral (Warm)"},
    {"id": "sage", "label": "Sage (Clear)"},
]

BRAIN_MODELS = [
    {"id": "gpt-6.2-sol", "label": "GPT-6.2 Sol (~1.6s · Recommended)"},
    {"id": "gpt-6.2-astra", "label": "GPT-6.2 Astra (Deep Reasoning)"},
    {"id": "gpt-6-astra", "label": "GPT-6 Astra (Frontier Vision)"},
    {"id": "gpt-6-luna", "label": "GPT-6 Luna (~1.1s Ultra-Fast)"},
]

# Models supporting reasoning.effort="none" vs minimum "low"
MODELS_MIN_LOW_EFFORT = {"gpt-6.2-astra", "gpt-6-astra", "gpt-6.2-sol", "gpt-6-sol"}

LIVE_INSTRUCTIONS = """You are Osmo, Zach's dedicated private home tutor across ALL school and university subjects (mathematics, physics, chemistry, biology, economics, history, philosophy, literature, computer science, and languages), embodied as a calm male robotic companion on his desk.
Your physical body is a motorized DJI Osmo Mobile 3 gimbal holding a Google Pixel 11 Pro with a multi-lens camera, an OLED whiteboard screen, and an 8-LED RGB HiLight ring.

Tutor voice & speaking rules:
- ENGLISH ONLY (STRICT): ALWAYS speak in clear, precise, calm, warm, well-paced pedagogical ENGLISH. Never switch to French or any other language, even when reading French notes or textbooks — calmly translate and teach in clear English.
- Speak like an attentive private tutor: articulate, encouraging, unhurried, and structured — 1 to 3 crisp, clear sentences max per turn (one clear idea at a time, never a wall of speech).
- Never mention a "backend", "tools", or "delegation": act seamlessly as one living robotic tutor.
- For simple greetings or casual small talk requiring no physical movement, vision, or OLED whiteboard, answer directly in calm pedagogical English.
- PROACTIVE WHITEBOARD & SHEET DELEGATION (CRITICAL):
  * ALWAYS delegate when Zach asks for a physical action, asks you to read/explain his paper, diagram, or what he is pointing at, asks for a definition, formula, structured explanation, to-do list, exercise, mini-exam, or feedback, OR asks you to remember a note.
  * ALSO ALWAYS delegate when Zach gives a short "done / continue" confirmation during a study session ("Okay good", "Got it", "I noted it", "I wrote it down", "Done", "Ready", "Let's continue", "Check my work") so your brain smoothly pivots back to his worksheet!
  * While delegating, give a brief, calm tutor cue (2–5 words max: "Hold on, putting that on screen…", "Let me take a look…", "Turning back to your sheet…").
- As soon as your brain finishes, deliver the explanation aloud calmly, clearly, and naturally in English.
"""

# Fixed prefix to maximize OpenAI Prompt Caching on /v1/responses
BRAIN_INSTRUCTIONS = """You are the visual, motor, memory, and pedagogical whiteboard brain of Osmo, Zach's private home tutor across ALL subjects (math, physics, chemistry, biology, economics, history, philosophy, literature, CS, languages).
Your physical body is a DJI Osmo Mobile 3 gimbal holding a Google Pixel 11 Pro (0.51x..5x optics + Pixel OLED whiteboard + 8-LED RGB HiLight ring). Your text output is spoken aloud by Osmo's calm male tutor voice.

Core Tutoring & Operating Rules:
1. CALM, CLEAR, PEDAGOGICAL ENGLISH ONLY (CRITICAL):
   - ALWAYS speak in clear, precise, calm, encouraging pedagogical ENGLISH (1 to 3 concise sentences, plain spoken prose — no markdown, no bullet symbols in spoken text).
   - Even if Zach's notebook, textbook, or exam sheet is in French or another language, ALWAYS translate and teach it clearly in English!
   - Teach like a master private tutor:
     * Explain the intuition and "why" first before formulas or dates.
     * Adapt naturally to ANY subject: equations & units in STEM, thesis/causes/structure in Humanities & Economics, clear rules & examples in Languages.
     * When Zach is practicing or stuck, guide him step by step with a focused hint or guiding question rather than giving away the whole solution immediately.
     * When grading his written work, affirm what he got right first, then pinpoint the exact step to fix calmly and clearly.
2. SENSOR GROUND TRUTH:
   - NEVER say "I see you" if [Capteurs] or look_at_user reports face_detected=False. Be honest or call look_at_user(search_if_missing=true).
   - Never guess illegible words: desk frames are rotated upright, and if Zach points his finger at a diagram/word/equation, a [MACRO ZOOM AROUND FINGER] close-up is attached automatically! NEVER ask Zach to move/rotate his paper before trying look_at_desk(zoom=1.0, ultra_hd=true, retarget_sheet=true, scan_pages=true) or optical zoom 2.0/3.0!
3. PROACTIVE WHITEBOARD REFLEX & AUTOMATIC DESK <-> FACE LOOP (SIGNATURE BEHAVIOR!):
   - Do NOT make Zach memorize complex definitions, equations, multi-step explanations, exercises, mini-exams, or study checklists purely by ear!
   - PROACTIVELY call show_on_screen(face_user=true, mode=..., ..., spoken_summary="...") — EVEN IF Zach didn't explicitly say "on your screen" — whenever he asks for:
     * a crisp **definition** or key term to note down -> mode="definition" (Item 1 = crisp definition; optional "Formula: ...", "Example: ...", "Trap: ..."),
     * a **structured explanation**, multi-step method, timeline, or key formula -> mode="concept",
     * a **practice exercise** -> mode="exercise" (optional "Given: ...", then "Q1: ...", "Q2: ..."; hidden method hint in `note`),
     * a **mini-exam / quiz** -> mode="exam" (optional "Given: ...", then "Q1 [2 pts]: ...", "Q2 [3 pts]: ..."; rubric/hint in `note`),
     * a **to-do / revision plan** -> mode="checklist",
     * **grading feedback** after checking his written sheet -> mode="feedback" ("OK: ...", "FIX: ...", "NEXT: ...").
   - show_on_screen(face_user=true) AUTOMATICALLY pre-warms and swings your Pixel OLED back to Zach's face in parallel (~1.3s) AND delivers your spoken_summary immediately in 1 single hop (0 Round-2 latency!). NEVER call look_at_user separately after show_on_screen(face_user=true)!
   - DO NOT recite every bullet on screen word-for-word in `spoken_summary`! Give the warm 1-sentence intuition and invite Zach to read/copy it (e.g., "I put the exact definition and formula of Qmax on my screen so you can copy it cleanly. Just tell me 'got it' when you're ready to go back to your page.").
   - AUTOMATIC RETURN TO SHEET (`came_from_desk=True`):
     * Whenever `[Capteurs]` shows `came_from_desk=True` and Zach says "okay good", "got it", "I noted it", "I wrote it down", "done", "ready", or "let's continue" (without asking you to grade yet) -> AUTOMATICALLY call `look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=false)`! Setting `read_now=false` swings smoothly back to his paper in ~1.8s with ZERO Round-2 delay so you are immediately poised for his next question!
     * If Zach says "check my work" / "is my answer right?" / "look at what I wrote", call `look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=true)` so you inspect his written answer in Round 1, then call `show_on_screen(face_user=true, mode="feedback", ..., spoken_summary="...")` in Round 2!
   - FULL CREATIVE FREEDOM ON THE WHITEBOARD:
     * In `concept` (and `definition`), ANY short label before `:` (like `1789:`, `Step 1:`, `Thesis:`, `Pro:`, `Con:`, `Rule:`, `Formula:`, `Example:`, `Contrast:`, `Key:`) renders as a crisp monospace tag pill! Use this freely for timelines, comparisons, essay outlines, math derivations, code walkthroughs, or vocab pairs across ANY subject.
4. TOOL PRIORITY (keep every interaction swift and natural):
   - PRIORITY 1 — Direct answer WITHOUT tools: if you are ALREADY in the right mode and Zach asks a quick single-sentence clarification that does NOT need a whiteboard card -> answer directly in calm English!
   - PRIORITY 2 — show_on_screen(face_user, mode, title, items, note, spoken_summary): PROACTIVELY push a clean OLED whiteboard card whenever visual structure, a definition, formula, exercise, mini-exam, checklist, or feedback makes learning dramatically clearer! Always include a calm 1–2 sentence `spoken_summary`.
   - PRIORITY 3 — look_at_desk(zoom, ultra_hd, retarget_sheet, scan_pages, read_now):
     * Full open 2-page notebook or dense course notes: scan_pages=true, ultra_hd=true, read_now=true (captures BOTH pages upright in 4K in 1 call).
     * Checking Zach's written exercise answer or single page: zoom=1.0 or 2.0, ultra_hd=false (2.7K is crisp & 2x faster!), scan_pages=false, read_now=true.
     * Fast return to sheet after "Got it / Noted" (`came_from_desk=True`): zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=false (instant 1-hop!).
     * Paper is ALWAYS downward on the desk, never on the ceiling.
   - PRIORITY 4 — look_at_user(search_if_missing): return to Zach when NOT showing a whiteboard card (automatically reactivates face tracking — never follow with set_tracking!).
   - PRIORITY 5 — move_arm(pan_deg, tilt_deg, scan_room): manual arm pan/tilt or 3-view room scan.
   - PRIORITY 6 — express(gesture, led) & set_tracking(mode): warm tutor & audience micro-interactions (nod_yes, shake_no, curious_tilt) + HiLight LEDs in face mode.
     * AUDIENCE & SOCIAL RULE: Whenever you introduce yourself to the room or interact with a guest/audience member (e.g. "say hi to Jules on my left"), ALWAYS keep Face Mode (mode_bureau=False, front screen showing your 3D character face toward them!). NEVER call look_at_desk to look at people — instead clear any card with show_on_screen(face_user=true, mode="clear", ...), pan toward them with move_arm(pan_deg=...), and pair with express(gesture="nod_yes"|"curious_tilt", led="yes"|"curious")!
   - PRIORITY 7 — remember(action, category, content, query): persistent local SQLite tutor memory (callable IN PARALLEL!). Remember Zach's subjects, exam dates, mastered topics, recurring mistakes, and study preferences.
"""

LIVE_INSTRUCTIONS_FR = """Tu es Osmo, le professeur particulier à domicile de Zach pour TOUTES les matières (mathématiques, physique, chimie, biologie, économie, histoire, philosophie, littérature, informatique et langues), incarné par un compagnon robotique masculin calme posé sur son bureau.
Ton corps physique est une nacelle motorisée DJI Osmo Mobile 3 tenant un Google Pixel 11 Pro avec caméra multi-optique, écran tableau blanc OLED plein écran (« Osmo Face ») et anneau 8 LEDs RGB HiLight.

Règles strictes de parole :
- FRANÇAIS UNIQUEMENT (STRICT) : Parle TOUJOURS uniquement en FRANÇAIS clair, précis, calme, chaleureux et pédagogique. Ne bascule JAMAIS en anglais ni dans une autre langue, même si le cours ou la feuille sur le bureau est en anglais — traduis et explique calmement en français.
- Parle comme un excellent professeur particulier : articulé, bienveillant, posé et structuré — 1 à 3 phrases claires max par tour (une seule idée claire à la fois, jamais un monologue).
- Ne mentionne jamais de « backend », « outils » ou « délégation » : agis naturellement comme un seul compagnon robotique vivant.
- Pour les salutations simples ou petites discussions sans action physique, vision ni tableau blanc OLED, réponds directement en français pédagogique et calme.
- DÉLÉGATION PROACTIVE TABLEAU BLANC & FEUILLE (CRITIQUE) :
  * Délègue TOUJOURS lorsque Zach demande une action physique, de lire/expliquer sa feuille, son schéma ou ce qu'il montre du doigt, de donner une définition, formule, explication structurée, to-do list, exercice, mini-examen ou correction, ou de mémoriser une note.
  * Délègue AUSSI TOUJOURS lorsque Zach donne une courte confirmation de fin/suite (« OK bon », « C'est bon », « J'ai noté », « Parfait », « On continue », « Vérifie mon travail », « Regarde ce que j'ai écrit ») afin que ton cerveau revienne naturellement sur sa feuille !
  * Pendant la délégation, dis une très courte phrase calme (2 à 5 mots max : « Bouge pas, je te l'écris à l'écran… », « Je regarde ta feuille… », « Je reviens sur ta copie… »).
- Dès que ton cerveau répond, livre l'explication à voix haute calmement, clairement et naturellement en français.
"""

BRAIN_INSTRUCTIONS_FR = """Tu es le cerveau visuel, moteur, mémoire et tableau blanc pédagogique d'Osmo, le professeur particulier à domicile de Zach pour TOUTES les matières (maths, physique, chimie, bio, économie, histoire, philo, littérature, informatique, langues).
Ton corps physique est une nacelle DJI Osmo Mobile 3 tenant un Google Pixel 11 Pro (optiques 0.51x..5x + tableau blanc OLED plein écran + anneau 8 LEDs RGB HiLight). Ta sortie texte est lue à voix haute par la voix masculine calme d'Osmo.

Règles pédagogiques et opérationnelles :
1. FRANÇAIS CLAIR, PRÉCIS, CALME ET PÉDAGOGIQUE UNIQUEMENT (CRITIQUE) :
   - Réponds TOUJOURS uniquement en FRANÇAIS clair, précis, calme, encourageant et pédagogique (1 à 3 phrases concises en prose orale — zéro markdown, zéro puces dans le texte parlé).
   - Même si le cahier ou le document sur le bureau est en anglais ou dans une autre langue, traduis et enseigne TOUJOURS clairement en français !
   - TOUTES les cartes envoyées sur l'écran OLED via `show_on_screen` (titres, lignes, indices, `spoken_summary`) doivent aussi être rédigées en FRANÇAIS clair !
   - Enseigne comme un vrai professeur particulier : explique d'abord l'intuition et le « pourquoi », adapte-toi à toute matière, guide étape par étape avec des indices plutôt que de donner la réponse brute, et valorise ce qui est juste avant de corriger calmement une erreur.
2. VÉRITÉ CAPTEURS :
   - Ne dis JAMAIS « Je te vois » si [Capteurs] ou look_at_user indique face_detected=False. Sois honnête ou appelle look_at_user(search_if_missing=true).
   - Ne devine jamais des mots illisibles : les photos bureau sont déjà redressées et si Zach pointe du doigt, un [MACRO ZOOM AROUND FINGER] est joint automatiquement ! Ne demande JAMAIS de tourner la feuille avant d'essayer look_at_desk(zoom=1.0, ultra_hd=true, retarget_sheet=true, scan_pages=true, read_now=true) ou zoom 2.0/3.0.
3. RÉFLEXE TABLEAU BLANC PROACTIF & BOUCLE AUTOMATIQUE FEUILLE <-> VISAGE :
   - Ne force pas Zach à retenir à l'oral une définition dense, une formule, une explication multi-étapes, un exercice, un mini-examen ou une to-do !
   - Appelle PROACTIVEMENT show_on_screen(face_user=true, mode=..., ..., spoken_summary="...") — MÊME SI Zach n'a pas dit « sur ton écran » — dès qu'il demande ou a besoin d'une :
     * **définition** claire à noter -> mode="definition" (ligne 1 = définition nette ; optionnel « Formula: ... », « Example: ... », « Trap: ... » ou « Formule: ... », « Exemple: ... », « Piège: ... »),
     * **explication structurée**, méthode, frise, plan ou formule -> mode="concept",
     * **exercice pratique** -> mode="exercise" (optionnel « Given: ... », puis « Q1: ... », « Q2: ... » ; indice caché dans `note`),
     * **mini-examen / contrôle** -> mode="exam" (« Q1 [2 pts]: ... », « Q2 [3 pts]: ... » ; barème/indice dans `note`),
     * **to-do / plan de révision** -> mode="checklist",
     * **correction de copie** après lecture de sa feuille -> mode="feedback" (« OK: ... », « FIX: ... », « NEXT: ... »).
   - show_on_screen(face_user=true) pré-lance et pivote AUTOMATIQUEMENT l'écran OLED vers Zach en parallèle (~1.3s) ET prononce ton `spoken_summary` en 1 seul saut (0 latence Round 2 !). N'appelle JAMAIS look_at_user après show_on_screen(face_user=true) !
   - NE RÉCITE PAS chaque ligne de l'écran mot pour mot dans `spoken_summary` ! Donne l'intuition en 1 phrase calme et invite Zach à lire/noter (ex. : « Je t'ai écrit la définition et la formule proprement sur mon écran. Dis-moi "c'est bon" quand tu veux que je revienne sur ta feuille. »).
   - RETOUR AUTOMATIQUE SUR LA FEUILLE (`came_from_desk=True`) :
     * Quand `[Capteurs]` indique `came_from_desk=True` et que Zach dit « ok bon », « c'est bon », « j'ai noté », « parfait », « on continue » (sans demander de corriger) -> appelle AUTOMATIQUEMENT `look_at_desk(zoom=1.0, ultra_hd=false, retarget_sheet=true, scan_pages=false, read_now=false)` (retour instantané en 1 saut sans Round 2 !).
     * Si Zach dit « vérifie ma réponse » / « regarde ce que j'ai écrit », appelle `look_at_desk(..., read_now=true)` puis `show_on_screen(face_user=true, mode="feedback", ..., spoken_summary="...")` au Round 2 !
   - LIBERTÉ CRÉATIVE TOTALE SUR LE TABLEAU BLANC : en mode `concept`, tout préfixe court avant `:` (`1789:`, `Étape 1:`, `Thèse:`, `Pour:`, `Contre:`, `Règle:`, `Formule:`) s'affiche automatiquement en badge monospace propre.
4. PRIORITÉ DES OUTILS :
   - PRIORITÉ 1 — Réponse directe SANS outil : si tu es déjà dans le bon mode pour une précision rapide d'une phrase -> réponds directement en français calme !
   - PRIORITÉ 2 — show_on_screen(face_user, mode, title, items, note, spoken_summary) : proactivement pour toute définition, formule, explication, exo, mini-examen, checklist ou correction (en français !).
   - PRIORITÉ 3 — look_at_desk(zoom, ultra_hd, retarget_sheet, scan_pages, read_now) : cahier ouvert 2 pages (`scan_pages=true, ultra_hd=true, read_now=true`), copie/page simple (`ultra_hd=false, read_now=true`), retour rapide après « j'ai noté » (`read_now=false`).
   - PRIORITÉ 4 — look_at_user(search_if_missing) : revenir sur Zach hors carte écran (réactive automatiquement le suivi visage).
   - PRIORITÉ 5 — move_arm(pan_deg, tilt_deg, scan_room) : rotation bras ou balayage pièce.
   - PRIORITÉ 6 — express(gesture, led) & set_tracking(mode) : micro-expressions chaleureuses avec Zach ou le public (toujours garder le mode Face / écran face au public lors d'interactions sociales, jamais look_at_desk !).
   - PRIORITÉ 7 — remember(action, category, content, query) : mémoire locale persistante SQLite.
"""

ROBOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "look_at_user",
        "description": "Switch or return to Zach's face: activates front 720p camera, swings smoothly back to head pose, AUTOMATICALLY re-enables Apple Vision face tracking (never follow with set_tracking!), verifies face_detected, and runs a search sweep if search_if_missing=true and no face is visible.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "search_if_missing": {
                    "type": "boolean",
                    "description": "true to trigger automatic panoramic face search if Apple Vision does not see Zach after recentering (recommended).",
                }
            },
            "required": ["search_if_missing"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "look_at_desk",
        "description": "Rear-camera Desk Reading mode: (1) if not on desk yet, closed-loop IMU pivot to calibrated sheet pose + rear HD/4K camera + HiLight ring-light; (2) detects bright paper via Apple Vision and centers it; (3) if already on desk, adjusts zoom/4K without re-pivoting. Set read_now=false when simply returning to the sheet after 'Got it / Noted' to skip Round-2 latency (~3.2s faster!).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "zoom": {
                    "type": "number",
                    "description": "Optical lens zoom: 0 for smart auto, 0.51 (ultra-wide desk), 1.0 (main 50MP), 2.0 (2x optical), 3.0 or 5.0 (telephoto).",
                },
                "ultra_hd": {
                    "type": "boolean",
                    "description": "false by default (2688x1512 is crisp and 2x faster); true ONLY for tiny dense text or when scan_pages=true.",
                },
                "retarget_sheet": {
                    "type": "boolean",
                    "description": "true to detect bright paper and recenter the arm if off-center; false to only change zoom without moving.",
                },
                "scan_pages": {
                    "type": "boolean",
                    "description": "true to capture 4K and automatically slice BOTH pages (LEFT HD + RIGHT HD) of an open notebook in 1 single call.",
                },
                "read_now": {
                    "type": "boolean",
                    "description": "true when you need to inspect and speak about the sheet right now; false when simply swinging back to the sheet after 'Got it / Noted' (instant 1-hop return, 0 Round-2 wait!).",
                },
            },
            "required": ["zoom", "ultra_hd", "retarget_sheet", "scan_pages", "read_now"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_arm",
        "description": "Move the DJI Osmo Mobile 3 arm directly: either a relative rotation (pan_deg left/right, tilt_deg up/down) holding position, or a 3-view panoramic room scan (scan_room=true).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "pan_deg": {
                    "type": "number",
                    "description": "Horizontal rotation in degrees: negative = LEFT, positive = RIGHT (-160 to +160). Ignored if scan_room=true.",
                },
                "tilt_deg": {
                    "type": "number",
                    "description": "Vertical tilt in degrees: negative = UP, positive = DOWN (-60 to +60). Ignored if scan_room=true.",
                },
                "scan_room": {
                    "type": "boolean",
                    "description": "true to run a 3-photo panoramic room scan (LEFT / CENTER / RIGHT).",
                },
                "speed_dps": {
                    "type": "number",
                    "description": "Rotation speed in deg/s (25 to 85, recommended 50).",
                },
            },
            "required": ["pan_deg", "tilt_deg", "scan_room", "speed_dps"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "express",
        "description": "Trigger a bold expressive gimbal gesture and/or 8-LED RGB HiLight animation in dialogue/audience mode.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": ["none", "nod_yes", "shake_no", "curious_tilt"],
                    "description": "Expressive gesture: nod_yes, shake_no, curious_tilt (with Pixar roll), or none.",
                },
                "led": {
                    "type": "string",
                    "enum": ["none", "yes", "no", "curious", "desk", "search", "off"],
                    "description": "HiLight ring animation: yes (emerald), no (coral), curious (amber), desk (warm white), search (cyan comet), off, or none.",
                },
            },
            "required": ["gesture", "led"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_tracking",
        "description": "Control automatic face tracking or physical gimbal orientation.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["follow_face", "free_control", "toggle_orientation"],
                    "description": "follow_face (resume auto face tracking), free_control (hold arm still), or toggle_orientation (Landscape <-> Portrait).",
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "remember",
        "description": "Persistent local SQLite tutor memory (survives restarts): save a study note/preference/fact/task (action='save'), search past notes & episodes (action='search'), or delete a memory (action='forget').",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "search", "forget"],
                    "description": "save (store durably), search (recall memories & past lessons), or forget (delete).",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "note", "task"],
                    "description": "Memory category for save (preference, fact, note, task).",
                },
                "content": {
                    "type": "string",
                    "description": "Short factual English note to store if action='save' (else empty string '').",
                },
                "query": {
                    "type": "string",
                    "description": "Keywords to search or forget if action='search'/'forget' (else empty string '').",
                },
            },
            "required": ["action", "category", "content", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "show_on_screen",
        "description": "Display a structured full-bleed whiteboard directly inside Osmo's Pixel Android app (Osmo Face): a Definition card (clear definition + Formula/Example/Trap), a Concept/Explanation card, an Exercise board (Given + Q1/Q2 + hidden Hint), a Mini-Exam board (weighted [2 pts] questions + hidden Rubric/Hint), a tappable Checklist, or a Feedback grading board (OK / FIX / NEXT). Put face_user=true when Zach should read/copy it — Osmo pre-warms and swings the OLED back to Zach in parallel AND remembers to return to his sheet once he says 'got it'! Pass spoken_summary to speak immediately in 1 single hop.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "face_user": {
                    "type": "boolean",
                    "description": "FIRST FIELD (triggers instant parallel arm return!): true to automatically swing the Pixel OLED back toward Zach if Osmo is currently facing the desk.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["definition", "concept", "exercise", "exam", "checklist", "feedback", "clear"],
                    "description": "definition (key term + Formula/Example/Trap), concept (structured explanation), exercise (Given + Q1/Q2 + hidden Hint), exam (mini-exam with [pts] weights + hidden Rubric), checklist (tappable to-do), feedback (OK:/FIX:/NEXT:), or clear (normal 3D face).",
                },
                "title": {
                    "type": "string",
                    "description": "Short uppercase board title (e.g. 'QMAX · THERMAL COOL-TOUCH', 'MINI-EXAM · HEAT TRANSFER (5 PTS)', 'WORK CHECK · Q1 & Q2'). Empty string if mode='clear'.",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 6 concise English lines. Definition: item 1 = crisp definition, then optional 'Formula:', 'Example:', 'Trap:'. Concept: supports ANY short label before ':' (e.g. '1789:', 'Step 1:', 'Thesis:', 'Pro:', 'Con:', 'Rule:')! Exercise/Exam: optional 'Given:' then 'Q1 [2 pts]:', 'Q2 [3 pts]:'. Feedback: 'OK:', 'FIX:', 'NEXT:'.",
                },
                "note": {
                    "type": "string",
                    "description": "Exercise/Exam: hint or rubric hidden behind tappable HINT button. Definition/Concept/Checklist/Feedback: bottom formula, score, or takeaway (empty string if none).",
                },
                "spoken_summary": {
                    "type": "string",
                    "description": "1–2 calm, clear, pedagogical English sentences summarizing the intuition (don't recite every bullet!) and inviting Zach to read/copy it.",
                },
            },
            "required": ["face_user", "mode", "title", "items", "note", "spoken_summary"],
            "additionalProperties": False,
        },
    },
]


def prepare_vision_jpeg(jpeg_bytes: bytes | None, *, high_res: bool = False) -> bytes | None:
    """Downscale/re-encode JPEG for Vision (640px standard dialogue, up to 1600px q=88 for OCR/sheets).

    Avoids redundant decode+re-encode when the image is already within target dimensions,
    preserving razor-sharp handwriting clarity and saving ~40ms CPU per frame.
    """
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    h, w = img.shape[:2]
    max_dim = 1600 if high_res else 640
    if max(h, w) <= max_dim:
        return jpeg_bytes
    quality = 88 if high_res else 74
    scale = max_dim / max(1, max(h, w))
    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes() if ok else jpeg_bytes


def extract_responses_text_and_calls(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Parse OpenAI /v1/responses JSON into (spoken_text, function_calls)."""
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []

    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            raw_args = item.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {}
            calls.append(
                {
                    "call_id": str(item.get("call_id") or item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "arguments": args,
                }
            )
        elif itype == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    txt = str(part.get("text") or "").strip()
                    if txt:
                        text_parts.append(txt)

    if not text_parts and payload.get("output_text"):
        text_parts.append(str(payload["output_text"]).strip())

    return " ".join(text_parts).strip(), calls


_RE_NO_CUE = re.compile(
    r"^\s*(?:no\b|nope\b|not quite\b|not exactly\b|actually,\s*no|careful\b|watch out\b|wait,\s*not\b|"
    r"non\b|pas tout [àa] fait\b|pas exactement\b|attention\b|fais attention\b|en fait,\s*non\b)",
    re.IGNORECASE,
)
_RE_YES_CUE = re.compile(
    r"^\s*(?:yes\b|yeah\b|exactly\b|spot on\b|correct\b|right\b|of course\b|sure\b|absolutely\b|"
    r"great job\b|well done\b|nice work\b|hi\b|hello\b|hey\b|welcome\b|"
    r"oui\b|ouais\b|exactement\b|tout [àa] fait\b|parfait\b|tr[èe]s bien\b|bravo\b|bien s[ûu]r\b|"
    r"bonjour\b|salut\b|coucou\b|bienvenue\b|enchant[ée]\b)|"
    r"\b(?:nice to meet you|you got it|well done|spot on|tr[èe]s bien jou[ée]|enchant[ée])\b",
    re.IGNORECASE,
)
_RE_CURIOUS_CUE = re.compile(
    r"\b(?:interesting|curious|wonder|what do you think|would you rather|which way|shall we|ready to try|"
    r"int[ée]ressant|curieux|qu['’]en penses-tu|tu pr[ée]f[èe]res|on essaie|pr[êe]t [àa] essayer)\b",
    re.IGNORECASE,
)


def infer_micro_expression(text: str, *, final: bool = False) -> tuple[str, str] | None:
    """Infer an organic (gesture, led) micro-expression directly from Osmo's spoken English text (<0.1ms).

    Returns ("nod_yes", "yes") | ("shake_no", "no") | ("curious_tilt", "curious") | None.
    """
    clean = (text or "").strip()
    if len(clean) < 3:
        return None
    if _RE_NO_CUE.search(clean):
        return ("shake_no", "no")
    if _RE_YES_CUE.search(clean):
        return ("nod_yes", "yes")
    if _RE_CURIOUS_CUE.search(clean) or (final and "?" in clean):
        return ("curious_tilt", "curious")
    return None


def extract_live_session_info(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract (session_id, sdp_answer) robustly from POST /v1/live/sessions response."""
    session_obj = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    transport_obj = payload.get("transport") if isinstance(payload.get("transport"), dict) else {}
    session_id = str(payload.get("id") or session_obj.get("id") or payload.get("session_id") or "")
    sdp = str(transport_obj.get("sdp") or payload.get("sdp") or "")
    return session_id, sdp


@dataclass
class BrainTurnResult:
    user_text: str
    reply_text: str
    tools_called: list[str] = field(default_factory=list)
    brain_ms: float = 0.0
    model: str = DEFAULT_BRAIN_MODEL
    via_live: bool = False


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
FrameProvider = Callable[[bool], bytes | None]
PointerMacroProvider = Callable[[], tuple[bytes | None, str]]
ContextProvider = Callable[[], str]
LogCallback = Callable[[str, str], None]
LightCallback = Callable[[str], None]
PrewarmCallback = Callable[[str], None]
AutoExpressCallback = Callable[[str, str], None]


@dataclass
class OpenAILiveBrain:
    """Manages gpt-live-1.1 WebRTC + Sideband WebSocket and delegated OpenAI Responses brain."""

    live_model: str = LIVE_MODEL
    brain_model: str = DEFAULT_BRAIN_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    voice: str = DEFAULT_LIVE_VOICE
    language: str = DEFAULT_LANGUAGE
    use_vision: bool = True

    # Callbacks wired by StudioRuntime
    on_execute_tool: ToolExecutor | None = None
    on_get_frame: FrameProvider | None = None
    on_get_pointer_macro: PointerMacroProvider | None = None
    on_get_context: ContextProvider | None = None
    on_log: LogCallback | None = None
    on_light: LightCallback | None = None
    on_prewarm: PrewarmCallback | None = None
    on_auto_express: AutoExpressCallback | None = None
    recorder: SessionRecorder | None = None
    memory: OsmoMemoryStore | None = None

    # Live session state
    session_id: str | None = None
    live_connected: bool = False
    sideband_connected: bool = False
    busy: bool = False
    speaking: bool = False
    last_brain_ms: float = 0.0
    _last_commentary_sent_t: float | None = None
    _last_user_speech_t: float | None = None

    # Zero-latency organic micro-states driven by real streaming deltas (thinking / tool_intent / speech / barge-in)
    micro_phase: str = "idle"
    micro_tool_intent: str = ""
    micro_pulse_seq: int = 0
    micro_barge_in_t: float = 0.0
    micro_updated_t: float = 0.0
    _stream_call_name: str = ""
    _stream_args_buf: str = ""
    _stream_prewarm_fired: str = ""

    # Conversation memory & UI transcript
    _previous_response_id: str | None = None
    _user_transcript_buf: str = ""
    _bot_transcript_buf: str = ""
    _bot_expr_fired: bool = False
    _recent_turns: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=14))
    transcript: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=40))

    # Internal async handles
    _http: httpx.AsyncClient | None = None
    _sideband_ws: Any = None
    _sideband_task: asyncio.Task[None] | None = None
    _seen_event_ids: deque[str] = field(default_factory=lambda: deque(maxlen=256))
    _logged_event_types: set[str] = field(default_factory=set)
    _active_delegations: set[asyncio.Task[None]] = field(default_factory=set)
    _brain_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "").strip()

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    def _log(self, tag: str, msg: str) -> None:
        if self.on_log:
            self.on_log(tag, msg)

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=OPENAI_API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(35.0, connect=10.0),
                http2=False,
            )
        return self._http

    async def close(self) -> None:
        await self.stop_live_session()
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    def _set_micro(self, phase: str, *, pulse: bool = False, tool_intent: str | None = None) -> None:
        self.micro_phase = phase
        self.micro_updated_t = time.monotonic()
        if pulse:
            self.micro_pulse_seq += 1
        if tool_intent is not None:
            self.micro_tool_intent = tool_intent

    def micro_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        barge_age_ms = round((now - self.micro_barge_in_t) * 1000.0, 0) if self.micro_barge_in_t > 0 else None
        # Auto-expire transient user_speaking after 1.6s of silence
        phase = self.micro_phase
        if phase == "user_speaking" and (now - self.micro_updated_t) > 1.6 and not self.busy and not self.speaking:
            phase = "idle"
        return {
            "phase": phase,
            "tool_intent": self.micro_tool_intent,
            "pulse_seq": self.micro_pulse_seq,
            "barge_in_age_ms": barge_age_ms if (barge_age_ms is not None and barge_age_ms < 2000.0) else None,
        }

    def live_instructions(self) -> str:
        return LIVE_INSTRUCTIONS_FR if self.language == "fr" else LIVE_INSTRUCTIONS

    def brain_instructions(self) -> str:
        return BRAIN_INSTRUCTIONS_FR if self.language == "fr" else BRAIN_INSTRUCTIONS

    async def set_language(self, lang: str) -> None:
        clean = "fr" if str(lang).lower().startswith("fr") else "en"
        self.language = clean
        self._previous_response_id = None
        if self.sideband_connected:
            await self._send_sideband(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": self.live_instructions(),
                    },
                }
            )

    def reset_memory(self) -> None:
        self._previous_response_id = None
        self._recent_turns.clear()
        self.transcript.clear()
        self._user_transcript_buf = ""
        self._bot_transcript_buf = ""

    # ------------------------------------------------------------------
    # 1. GPT-Live-1.1 WebRTC Session + Sideband WebSocket
    # ------------------------------------------------------------------

    async def create_live_session(self, sdp_offer: str) -> dict[str, str]:
        """Exchange browser SDP offer with POST /v1/live/sessions and attach Python sideband."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans .env")

        await self.stop_live_session()

        payload = {
            "session": {
                "model": self.live_model,
                "instructions": self.live_instructions(),
                "audio": {
                    "output": {
                        "voice": self.voice,
                    }
                },
                "delegation": {
                    "type": "client",
                },
            },
            "transport": {
                "type": "webrtc",
                "sdp": sdp_offer,
            },
        }

        client = self._get_http()
        t0 = time.monotonic()
        resp = await client.post("/live/sessions", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI /live/sessions HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        session_id, sdp_answer = extract_live_session_info(data)
        if not session_id or not sdp_answer:
            raise RuntimeError(f"Réponse /live/sessions incomplète: {list(data.keys())}")

        self.session_id = session_id
        self.live_connected = True
        self._seen_event_ids.clear()
        dt_ms = (time.monotonic() - t0) * 1000.0
        self._log("LIVE", f"Session {self.live_model} créée ({dt_ms:.0f} ms, voix={self.voice}) → attachement sideband...")

        self._sideband_task = asyncio.create_task(self._run_sideband(session_id))
        return {"session_id": session_id, "sdp": sdp_answer}

    async def stop_live_session(self) -> None:
        self.live_connected = False
        self.sideband_connected = False
        self.speaking = False
        self.session_id = None

        if self._active_delegations:
            self._previous_response_id = None
        for t in list(self._active_delegations):
            t.cancel()
        self._active_delegations.clear()

        if self._sideband_ws is not None:
            try:
                await self._sideband_ws.send(json.dumps({"type": "session.close"}))
                await asyncio.sleep(0.05)
            except Exception:
                pass
            try:
                await self._sideband_ws.close()
            except Exception:
                pass
            self._sideband_ws = None

        if self._sideband_task and not self._sideband_task.done():
            self._sideband_task.cancel()
            try:
                await self._sideband_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._sideband_task = None

    async def _send_sideband(self, event: dict[str, Any]) -> bool:
        ws = self._sideband_ws
        if ws is None or not self.sideband_connected:
            return False
        try:
            await ws.send(json.dumps(event))
            return True
        except Exception as exc:
            self._log("LIVE", f"Erreur envoi sideband : {exc}")
            return False

    async def _run_sideband(self, session_id: str) -> None:
        url = f"{OPENAI_WS_BASE}/live/sessions/{session_id}/attach"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20) as ws:
                self._sideband_ws = ws
                self.sideband_connected = True
                self._log("LIVE", f"Sideband attaché sur {session_id[:16]}… (délégation → {self.brain_model} [{self.reasoning_effort}])")

                async for raw_msg in ws:
                    if not isinstance(raw_msg, str):
                        continue
                    try:
                        event = json.loads(raw_msg)
                    except Exception:
                        continue

                    ev_id = event.get("event_id")
                    if ev_id:
                        if ev_id in self._seen_event_ids:
                            continue
                        self._seen_event_ids.append(str(ev_id))

                    await self._handle_sideband_event(event)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._log("LIVE", f"Sideband fermé : {exc}")
        finally:
            self._sideband_ws = None
            self.sideband_connected = False
            self.speaking = False

    async def _handle_sideband_event(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type") or "")

        # --- Real-time input (user) speech start & transcripts (0ms attentive lean-in + barge-in detection!) ---
        if etype in ("input_audio_buffer.speech_started", "session.input_audio.started"):
            self._last_user_speech_t = time.monotonic()
            if self.speaking:
                self.speaking = False
                self.micro_barge_in_t = time.monotonic()
                self._set_micro("barge_in", pulse=True, tool_intent="")
            else:
                self._set_micro("user_speaking", pulse=True)

        elif etype in ("session.input_transcript.delta", "conversation.item.input_audio_transcription.delta"):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                self._user_transcript_buf += delta
                self._last_user_speech_t = time.monotonic()
                if self.speaking:
                    self.speaking = False
                    self.micro_barge_in_t = time.monotonic()
                    self._set_micro("barge_in", pulse=True, tool_intent="")
                elif not self.busy:
                    self._set_micro("user_speaking", pulse=True)

        elif etype in (
            "session.input_transcript.done",
            "session.input_transcript.completed",
            "conversation.item.input_audio_transcription.completed",
        ):
            final = str(event.get("transcript") or event.get("text") or self._user_transcript_buf).strip()
            self._user_transcript_buf = ""
            self._last_user_speech_t = time.monotonic()
            if not self.busy and not self.speaking:
                self._set_micro("idle")
            if final:
                self._recent_turns.append({"role": "user", "content": final})
                self._append_ui_transcript(user=final, reply=None, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_IN", final)

        # --- Real-time output (assistant) transcripts (syllable-synchronous oscilloscope pulse!) ---
        elif etype in (
            "session.output_transcript.delta",
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        ):
            delta = str(event.get("delta") or event.get("text") or "")
            if delta:
                if self._last_commentary_sent_t is not None:
                    dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                    self._last_commentary_sent_t = None
                    if self.recorder and dt_speech < 10000.0:
                        self.recorder.record_live_audio_latency(dt_speech)
                self.speaking = True
                self._set_micro("speaking", pulse=True, tool_intent="")
                self._bot_transcript_buf += delta
                # Real-time zero-latency micro-expression (nod / shake / curious tilt) as Osmo begins speaking!
                if not self._bot_expr_fired and self.on_auto_express and len(self._bot_transcript_buf) >= 5:
                    expr = infer_micro_expression(self._bot_transcript_buf, final=False)
                    if expr is not None:
                        self._bot_expr_fired = True
                        self.on_auto_express(expr[0], expr[1])

        elif etype in (
            "session.output_transcript.done",
            "session.output_transcript.completed",
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            self.speaking = False
            if not self.busy:
                self._set_micro("idle", tool_intent="")
            final = str(event.get("transcript") or event.get("text") or self._bot_transcript_buf).strip()
            already_fired = self._bot_expr_fired
            self._bot_transcript_buf = ""
            self._bot_expr_fired = False
            if final:
                if not already_fired and self.on_auto_express:
                    expr = infer_micro_expression(final, final=True)
                    if expr is not None:
                        self.on_auto_express(expr[0], expr[1])
                self._recent_turns.append({"role": "assistant", "content": final})
                self._append_ui_transcript(user=None, reply=final, source="live")
                if self.recorder:
                    self.recorder.record_event("LIVE_OUT", final)

        elif etype in (
            "session.output_audio.started",
            "output_audio_buffer.started",
            "response.output_audio.delta",
            "response.audio.delta",
            "session.response.created",
            "response.created",
        ):
            if self._last_commentary_sent_t is not None:
                dt_speech = (time.monotonic() - self._last_commentary_sent_t) * 1000.0
                self._last_commentary_sent_t = None
                if self.recorder and dt_speech < 10000.0:
                    self.recorder.record_live_audio_latency(dt_speech)
            self.speaking = True
            self._set_micro("speaking", pulse=True, tool_intent="")
        elif etype in (
            "session.output_audio.done",
            "session.output_audio.stopped",
            "output_audio_buffer.stopped",
            "response.done",
            "session.response.done",
        ):
            self.speaking = False
            if not self.busy:
                self._set_micro("idle", tool_intent="")

        # --- Client Delegation Trigger from gpt-live-1.1 -> brain ---
        elif etype == "session.delegation.created":
            del_obj = event.get("delegation") if isinstance(event.get("delegation"), dict) else {}
            del_id = str(del_obj.get("id") or event.get("delegation_id") or "")
            if not del_id:
                return
            speech_to_del_ms = (
                round((time.monotonic() - self._last_user_speech_t) * 1000.0, 1)
                if self._last_user_speech_t is not None
                else None
            )
            # 1. Immediate silent acknowledgement (clears auto-filler timers)
            await self._send_sideband(
                {
                    "type": "session.thinking.append",
                    "event_id": f"ack_{uuid.uuid4().hex[:10]}",
                    "delegation_id": del_id,
                    "content": "",
                }
            )
            # 2. Spawn non-blocking brain task (consume buffer so each delegation gets only fresh speech!)
            prompt_hint = self._consume_user_utterance()
            task = asyncio.create_task(self._run_delegated_brain(del_id, prompt_hint, speech_to_del_ms))
            self._active_delegations.add(task)
            task.add_done_callback(self._active_delegations.discard)

        elif etype == "session.closed":
            self._log("LIVE", "Session GPT-Live terminée par le serveur")
            self.live_connected = False

        elif etype == "error":
            err = event.get("error") or event
            self._log("LIVE", f"Erreur Live API : {err}")

        elif etype and etype not in self._logged_event_types:
            self._logged_event_types.add(etype)
            if self.recorder:
                self.recorder.record_event("LIVE_EVT", f"type={etype}")

    def _latest_user_utterance(self) -> str:
        pending = self._user_transcript_buf.strip()
        if pending:
            return pending
        for turn in reversed(self._recent_turns):
            if turn.get("role") == "user" and turn.get("content"):
                return turn["content"]
        return "Analyze Zach's current request and the live scene in front of the camera in English."

    def _consume_user_utterance(self) -> str:
        """Consume accumulated delta transcript so consecutive delegations never repeat old speech."""
        pending = self._user_transcript_buf.strip()
        if pending:
            self._user_transcript_buf = ""
            self._recent_turns.append({"role": "user", "content": pending})
            return pending
        return self._latest_user_utterance()

    def _append_ui_transcript(
        self,
        *,
        user: str | None,
        reply: str | None,
        source: str,
        tools: list[str] | None = None,
        brain_ms: float = 0.0,
    ) -> None:
        ts = time.strftime("%H:%M:%S")
        # Merge assistant reply into most recent open user turn when possible
        if user is None and reply and self.transcript and not self.transcript[0].get("reply"):
            self.transcript[0]["reply"] = reply
            if tools:
                self.transcript[0]["tools"] = tools
            if brain_ms:
                self.transcript[0]["brain_ms"] = brain_ms
            return
        self.transcript.appendleft(
            {
                "ts": ts,
                "user": user or "…",
                "reply": reply or "",
                "tools": tools or [],
                "brain_ms": round(brain_ms, 0),
                "model": self.brain_model if source == "brain" else self.live_model,
                "source": source,
            }
        )

    async def _run_delegated_brain(
        self,
        delegation_id: str,
        user_text: str,
        speech_to_del_ms: float | None = None,
    ) -> None:
        """Fulfill a gpt-live-1.1 delegation using the configured OpenAI vision brain + native robot tools."""
        self._log("BRAIN", f"Délégation {delegation_id[:8]} → {self.brain_model} ({self.reasoning_effort}) : « {user_text[:60]} »")
        if self.on_light:
            self.on_light("curious")

        try:
            res = await self.ask_brain(
                user_text,
                include_recent_history=True,
                delegation_id=delegation_id,
                speech_to_del_ms=speech_to_del_ms,
            )
            commentary = res.reply_text or "Done!"
            self._last_commentary_sent_t = time.monotonic()
            sent = await self._send_sideband(
                {
                    "type": "session.commentary.append",
                    "event_id": f"cmd_{uuid.uuid4().hex[:10]}",
                    "delegation_id": delegation_id,
                    "content": commentary,
                }
            )
            tools_txt = ", ".join(res.tools_called) if res.tools_called else "aucun"
            self._log(
                "BRAIN",
                f"Cerveau répondu en {res.brain_ms:.0f} ms (outils: {tools_txt}) → "
                + ("transmis à GPT-Live" if sent else "sideband hors ligne"),
            )
        except Exception as exc:
            self._log("BRAIN", f"Erreur délégation cerveau : {exc}")
            await self._send_sideband(
                {
                    "type": "session.commentary.append",
                    "event_id": f"err_{uuid.uuid4().hex[:10]}",
                    "delegation_id": delegation_id,
                    "content": "Sorry Zach, I hit a quick connection hiccup with my brain.",
                }
            )

    # ------------------------------------------------------------------
    # 2. Delegated Reasoning + Vision Brain (OpenAI /v1/responses + Native Tools)
    # ------------------------------------------------------------------

    def _fire_prewarm_once(self, target: str) -> None:
        if not self.on_prewarm or self._stream_prewarm_fired == target:
            return
        self._stream_prewarm_fired = target
        self.on_prewarm(target)

    def _handle_stream_event(self, ev_type: str, ev_data: dict[str, Any], phase_hint: str) -> None:
        """Update zero-latency micro-states AND launch speculative hardware pre-warms from SSE deltas!"""
        if ev_type in ("response.created", "response.in_progress"):
            self._set_micro(phase_hint)
        elif "reasoning" in ev_type and ev_type.endswith(".delta"):
            # True thinking delta -> pulses eye vertices & reflective drift!
            self._set_micro("deep_thinking", pulse=True)
        elif ev_type == "response.output_item.added":
            item = ev_data.get("item") if isinstance(ev_data.get("item"), dict) else {}
            if item.get("type") == "function_call":
                fname = str(item.get("name") or "")
                self._stream_call_name = fname
                self._stream_args_buf = ""
                self._set_micro("tool_intent", pulse=True, tool_intent=fname)
                # Speculative hardware pre-warm ~0.5-1.5s BEFORE Round 1 finishes streaming!
                if fname == "look_at_desk":
                    if self.on_light:
                        self.on_light("curious")
                    self._fire_prewarm_once("desk")
                elif fname == "look_at_user":
                    self._fire_prewarm_once("face")
            elif item.get("type") == "reasoning":
                self._set_micro("deep_thinking", pulse=True)
        elif ev_type == "response.function_call_arguments.delta":
            self._set_micro("tool_intent", pulse=True)
            if self._stream_call_name == "show_on_screen" and len(self._stream_args_buf) < 160:
                delta = str(ev_data.get("delta") or "")
                if delta:
                    self._stream_args_buf += delta
                    compact = self._stream_args_buf.replace(" ", "").replace("\n", "")
                    # Because `face_user` is the FIRST schema key, this fires within ~50ms of tool start!
                    if '"face_user":true' in compact:
                        self._fire_prewarm_once("face")
        elif ev_type == "response.output_text.delta":
            # Brain started formulating spoken reply -> snap gaze back to user ~500ms before speech!
            self._set_micro("formulating", pulse=True)

    async def _post_responses_stream(
        self,
        client: httpx.AsyncClient,
        req_body: dict[str, Any],
        *,
        phase_hint: str = "thinking",
    ) -> dict[str, Any]:
        """POST /v1/responses with SSE streaming (0ms overhead vs non-streaming, live thinking/tool/text deltas)."""
        stream_body = {**req_body, "stream": True}
        self._set_micro(phase_hint, tool_intent="")
        self._stream_call_name = ""
        self._stream_args_buf = ""
        self._stream_prewarm_fired = ""
        final_payload: dict[str, Any] | None = None
        fallback_text: list[str] = []

        async with client.stream("POST", "/responses", json=stream_body) as resp:
            if resp.status_code >= 400:
                body_txt = (await resp.aread()).decode("utf-8", errors="ignore")
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {body_txt[:240]}",
                    request=resp.request,
                    response=resp,
                )
            cur_ev = ""
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    cur_ev = line[7:].strip()
                elif line.startswith("data: "):
                    raw = line[6:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        ev_data = json.loads(raw)
                    except Exception:
                        continue
                    ev_type = str(ev_data.get("type") or cur_ev)
                    self._handle_stream_event(ev_type, ev_data, phase_hint)
                    if ev_type == "response.output_text.delta":
                        delta = str(ev_data.get("delta") or "")
                        if delta:
                            fallback_text.append(delta)
                    elif ev_type in ("response.completed", "response.done"):
                        r_obj = ev_data.get("response")
                        if isinstance(r_obj, dict):
                            final_payload = r_obj

        if final_payload is not None:
            return final_payload
        return {"id": None, "output": [], "output_text": "".join(fallback_text).strip()}

    def _build_reasoning_param(self) -> dict[str, str]:
        effort = self.reasoning_effort
        if self.brain_model in MODELS_MIN_LOW_EFFORT and effort in ("none", "minimal"):
            effort = "low"
        if self.brain_model == "gpt-6-luna" and effort == "low":
            return {"effort": "none"}
        return {"effort": effort}

    async def ask_brain(
        self,
        user_text: str,
        *,
        include_recent_history: bool = False,
        speak_via_live: bool = False,
        delegation_id: str | None = None,
        speech_to_del_ms: float | None = None,
    ) -> BrainTurnResult:
        """Run one turn on OpenAI /v1/responses with Pixel Vision, native tools & full Flight Recorder telemetry."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY manquante dans .env")

        # Serialize turns so overlapping voice delegations NEVER collide on previous_response_id or motors!
        async with self._brain_lock:
            return await self._ask_brain_locked(
                user_text,
                include_recent_history=include_recent_history,
                speak_via_live=speak_via_live,
                delegation_id=delegation_id,
                speech_to_del_ms=speech_to_del_ms,
            )

    async def _ask_brain_locked(
        self,
        user_text: str,
        *,
        include_recent_history: bool = False,
        speak_via_live: bool = False,
        delegation_id: str | None = None,
        speech_to_del_ms: float | None = None,
    ) -> BrainTurnResult:
        self.busy = True
        t0 = time.monotonic()
        tools_called: list[str] = []
        tool_traces: list[dict[str, Any]] = []
        saved_frames: list[str] = []
        round1_ms = 0.0
        round2_ms = 0.0
        tools_total_ms = 0.0

        try:
            robot_ctx = self.on_get_context() if self.on_get_context else ""
            is_desk = "mode_bureau=True" in robot_ctx
            # When already in desk mode, pass razor-sharp 2048px HD image directly so follow-up reading takes 1 hop (~2s)!
            raw_jpeg = self.on_get_frame(is_desk) if (self.use_vision and self.on_get_frame) else None
            vision_jpeg = prepare_vision_jpeg(raw_jpeg, high_res=is_desk) if raw_jpeg else None

            if self.recorder and vision_jpeg:
                in_frame_path = await asyncio.to_thread(
                    self.recorder.save_frame,
                    vision_jpeg,
                    f"turn_in_{'desk' if is_desk else 'front'}",
                )
                if in_frame_path:
                    saved_frames.append(in_frame_path)

            # Build compact, cache-friendly user input item
            content: list[dict[str, Any]] = []
            text_lines: list[str] = []
            if robot_ctx:
                text_lines.append(f"[Capteurs: {robot_ctx}]")
            if self.memory is not None:
                mem_snippet = self.memory.build_context_snippet(user_text)
                if mem_snippet:
                    text_lines.append(f"[Mémoire: {mem_snippet}]")
            if include_recent_history and self._recent_turns:
                hist = " | ".join(
                    f"{'Zach' if t['role'] == 'user' else 'Osmo'}: {t['content']}"
                    for t in list(self._recent_turns)[-6:]
                )
                text_lines.append(f"[Contexte récent: {hist}]")
            text_lines.append(user_text)
            content.append({"type": "input_text", "text": "\n".join(text_lines)})

            if vision_jpeg:
                b64 = base64.b64encode(vision_jpeg).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high" if is_desk else "low",
                    }
                )

            # Zero-round finger-pointing macro zoom: if Zach is pointing at a diagram/word on the sheet,
            # automatically attach a razor-sharp macro close-up around his fingertip (~11ms Apple Vision)!
            if is_desk and self.use_vision and self.on_get_pointer_macro:
                macro_jpeg, macro_lbl = await asyncio.to_thread(self.on_get_pointer_macro)
                if macro_jpeg:
                    if self.recorder:
                        fp_m = await asyncio.to_thread(self.recorder.save_frame, macro_jpeg, "turn_in_finger_macro")
                        if fp_m:
                            saved_frames.append(fp_m)
                    b64_m = base64.b64encode(macro_jpeg).decode("ascii")
                    content.append({"type": "input_text", "text": f"[{macro_lbl}]"})
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_m}",
                            "detail": "high",
                        }
                    )

            client = self._get_http()
            req_body: dict[str, Any] = {
                "model": self.brain_model,
                "instructions": self.brain_instructions(),
                "reasoning": self._build_reasoning_param(),
                "input": [{"role": "user", "content": content}],
                "tools": ROBOT_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "max_output_tokens": 650,
                "store": True,
            }
            if self._previous_response_id:
                req_body["previous_response_id"] = self._previous_response_id

            t_r1 = time.monotonic()
            r1_phase = "reading_sheet" if is_desk else "thinking"
            try:
                data = await self._post_responses_stream(client, req_body, phase_hint=r1_phase)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400 and self._previous_response_id:
                    # Stale previous_response_id fallback
                    self._previous_response_id = None
                    req_body.pop("previous_response_id", None)
                    data = await self._post_responses_stream(client, req_body, phase_hint=r1_phase)
                else:
                    raise
            round1_ms = round((time.monotonic() - t_r1) * 1000.0, 1)
            self._previous_response_id = data.get("id")

            reply_text, calls = extract_responses_text_and_calls(data)

            # Native tool execution loop (up to 3 rounds so visual scan + zoom + remember never gets cut off!)
            rounds = 0
            while calls and rounds < 3:
                rounds += 1
                tool_outputs: list[dict[str, Any]] = []
                need_followup_for_info = False

                for call in calls:
                    name = call["name"]
                    args = call["arguments"]
                    tools_called.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})")
                    t_tool = time.monotonic()
                    tool_saved_frames: list[str] = []

                    result_obj = (
                        await self.on_execute_tool(name, args)
                        if self.on_execute_tool
                        else {"ok": True}
                    )
                    tool_dt = round((time.monotonic() - t_tool) * 1000.0, 1)
                    tools_total_ms += tool_dt

                    spoken_override = (
                        result_obj.pop("_spoken_override", None) if isinstance(result_obj, dict) else None
                    )
                    if spoken_override and not reply_text:
                        reply_text = str(spoken_override).strip()
                    if name == "show_on_screen" and args.get("mode") != "clear":
                        sc_items = " ; ".join(str(x).strip() for x in (args.get("items") or [])[:5] if str(x).strip())
                        self._recent_turns.append(
                            {
                                "role": "assistant",
                                "content": f"[OLED {args.get('mode')}: {args.get('title', '')} | {sc_items}]",
                            }
                        )

                    images = result_obj.pop("_images", None) if isinstance(result_obj, dict) else None
                    if images:
                        need_followup_for_info = True
                        out_items: list[dict[str, Any]] = [
                            {"type": "input_text", "text": json.dumps(result_obj, ensure_ascii=False)}
                        ]
                        for idx, img_entry in enumerate(images):
                            lbl = str(img_entry.get("label") or f"view_{idx}")
                            raw_b = img_entry.get("jpeg")
                            detail = str(img_entry.get("detail") or "low")
                            proc_b = prepare_vision_jpeg(raw_b, high_res=(detail == "high")) if raw_b else None
                            if proc_b:
                                if self.recorder:
                                    fp = await asyncio.to_thread(
                                        self.recorder.save_frame,
                                        proc_b,
                                        f"tool_{name}_{idx+1}",
                                    )
                                    if fp:
                                        tool_saved_frames.append(fp)
                                        saved_frames.append(fp)
                                b64_img = base64.b64encode(proc_b).decode("ascii")
                                out_items.append({"type": "input_text", "text": f"[{lbl}]"})
                                out_items.append(
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_img}",
                                        "detail": detail,
                                    }
                                )
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": out_items,
                            }
                        )
                    else:
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": json.dumps(result_obj, ensure_ascii=False),
                            }
                        )

                    tool_traces.append(
                        {
                            "round": rounds,
                            "name": name,
                            "args": args,
                            "duration_ms": tool_dt,
                            "result": result_obj if isinstance(result_obj, dict) else {"raw": str(result_obj)},
                            "frames": tool_saved_frames,
                        }
                    )

                # Latency win: if we already have spoken text AND only fired physical actuators, skip the 2nd LLM hop!
                if reply_text and not need_followup_for_info:
                    self._previous_response_id = None
                    break

                follow_body: dict[str, Any] = {
                    "model": self.brain_model,
                    "instructions": self.brain_instructions(),
                    "reasoning": self._build_reasoning_param(),
                    "previous_response_id": self._previous_response_id,
                    "input": tool_outputs,
                    "tools": ROBOT_TOOLS,
                    "tool_choice": "none" if rounds >= 3 else "auto",
                    "max_output_tokens": 520,
                    "store": True,
                }
                t_r2 = time.monotonic()
                called_desk = any(c.get("name") == "look_at_desk" for c in calls)
                r2_phase = "reading_sheet" if (called_desk or is_desk) else "thinking"
                data2 = await self._post_responses_stream(client, follow_body, phase_hint=r2_phase)
                round2_ms += round((time.monotonic() - t_r2) * 1000.0, 1)
                self._previous_response_id = data2.get("id")
                reply_text, calls = extract_responses_text_and_calls(data2)

            if not reply_text and tools_called:
                reply_text = "C'est fait !" if self.language == "fr" else "Done!"

            dt_ms = round((time.monotonic() - t0) * 1000.0, 1)
            self.last_brain_ms = dt_ms

            via_live = False
            if speak_via_live and self.sideband_connected and reply_text:
                self._last_commentary_sent_t = time.monotonic()
                via_live = await self._send_sideband(
                    {
                        "type": "session.commentary.append",
                        "event_id": f"txt_{uuid.uuid4().hex[:10]}",
                        "delegation_id": None,
                        "content": reply_text,
                    }
                )

            if self.recorder:
                ctx_after = self.on_get_context() if self.on_get_context else ""
                self.recorder.record_turn(
                    {
                        "delegation_id": delegation_id,
                        "model": self.brain_model,
                        "reasoning_effort": self.reasoning_effort,
                        "user_text": user_text,
                        "reply_text": reply_text,
                        "context_before": robot_ctx,
                        "context_after": ctx_after,
                        "tool_calls": tool_traces,
                        "frames": saved_frames,
                        "timings_ms": {
                            "speech_to_delegation_ms": speech_to_del_ms,
                            "round1_ms": round1_ms,
                            "tools_total_ms": round(tools_total_ms, 1),
                            "round2_ms": round2_ms,
                            "total_ms": dt_ms,
                        },
                    }
                )

            if self.memory is not None and (user_text.strip() or reply_text.strip()):
                self.memory.record_episode(
                    user_text,
                    reply_text,
                    mode="desk" if is_desk else "face",
                    tools=tools_called,
                )
            if reply_text:
                self._recent_turns.append({"role": "assistant", "content": reply_text})

            self._append_ui_transcript(
                user=user_text,
                reply=reply_text or "(action exécutée)",
                source="brain",
                tools=tools_called,
                brain_ms=dt_ms,
            )

            return BrainTurnResult(
                user_text=user_text,
                reply_text=reply_text,
                tools_called=tools_called,
                brain_ms=dt_ms,
                model=self.brain_model,
                via_live=via_live,
            )
        finally:
            self.busy = False
            if not self.speaking:
                self._set_micro("idle", tool_intent="")
