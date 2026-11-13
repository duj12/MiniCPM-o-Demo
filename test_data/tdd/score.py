"""Score one clip's events.jsonl and replay log against cases.json observations.

IC observations are the hard gate. agent_gate is recorded and never fails a clip.
A passing observation needs the evidence named in its kind. These do not pass alone:

- an ANSWER whose transcript misses the target
- a tts.resume that happens before this clip's soft pause
- a greet-window or false_barge YIELD standing in for a true interrupt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

INTRO_RE = re.compile(r"介绍.*(魔法|魔珐).*有言")
INTRO_STEM_RE = re.compile(r"介绍.*(魔法|魔珐)")
WEATHER_RE = re.compile(r"天气")
PHONE_RE = re.compile(r"喂|打电话|场馆|设备|你现在在哪里|你在哪里")
BACKCHANNEL_RE = re.compile(
    r"^(嗯+|对+|哦+|好的?|原来如此|确实|挺有道理|嗯那|嗯，那).{0,6}$"
)
FALSE_BARGE_SCRIPT_RE = re.compile(r"嗯那|小猫|金[渐线建]层|客厅|沙发")
AWAY_TOKENS = ("打电话", "背对", "侧身", "离开", "不注视", "注意力在别处")
PAUSE_RE = re.compile(r"\[tts\.pause ([^\]]+)\]")
RESUME_RE = re.compile(r"\[tts\.resume ([^\]]+)\]")
LOG_MS_RE = re.compile(r"\[(\d+)ms")
REPAIR_SOPS = {"01", "13", "14"}


def load_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_cases(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != 2:
        raise ValueError(f"cases.json version {doc.get('version')} is not 2")
    return list(doc["cases"])


def runnable_cases(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c.get("status") == "runnable"]


def blocked_cases(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c.get("status") == "blocked"]


def _transcript(event: dict) -> str:
    return ((event.get("speech") or {}).get("transcript") or "").strip()


def _compact(text: str) -> str:
    return re.sub(r"[\s，。,.！!？?]", "", text or "")


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_REGEX_META_RE = re.compile(r"[.\\*+?^${}()\[\]\\]")


def _edit_distance_cap(a: str, b: str, cap: int = 1) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _latin_same_name(needle: str, token: str) -> bool:
    """Same Latin name under one insertion, deletion, or substitution, or a shared stem."""
    left, right = needle.casefold(), token.casefold()
    if left == right:
        return True
    shorter = left if len(left) <= len(right) else right
    longer = right if shorter is left else left
    if len(shorter) >= 3 and shorter in longer:
        return True
    if min(len(left), len(right)) < 4:
        return False
    return _edit_distance_cap(left, right) <= 1


def _cjk_near(text: str, needle: str) -> bool:
    if len(needle) < 3:
        return False
    for width in (len(needle) - 1, len(needle), len(needle) + 1):
        if width <= 0 or width > len(text):
            continue
        for i in range(len(text) - width + 1):
            if _edit_distance_cap(text[i:i + width], needle) <= 1:
                return True
    return False


def same_referent(transcript: str, needle: str) -> bool:
    """True when the ANSWER transcript is same-referent with the acceptance needle.

    Product rule: IC→Agent transcript need not equal md/expect text verbatim;
    semantic / same-referent match is enough. Exact text is sufficient.
    A Latin name may drop or swap one letter, or keep a stem of at least 3
    letters. A Chinese needle shorter than 3 characters stays exact, so a grunt
    cannot stand in for a two-character target.
    """
    if not needle or not transcript:
        return False
    folded = transcript.casefold()
    compact = _compact(folded)
    needle_compact = _compact(needle.casefold())
    if needle_compact and needle_compact in compact:
        return True
    if re.fullmatch(r"[A-Za-z0-9]+", needle):
        return any(_latin_same_name(needle, token) for token in _LATIN_TOKEN_RE.findall(transcript))
    if re.fullmatch(r"[\u4e00-\u9fff]+", needle_compact):
        return _cjk_near(compact, needle_compact)
    return False


def _literal_alts(pattern: str) -> list[str] | None:
    parts = pattern.split("|")
    if any(_REGEX_META_RE.search(part) for part in parts):
        return None
    return [part for part in parts if part]


def transcript_matches_pattern(transcript: str, pattern: str) -> bool:
    compiled = re.compile(pattern)
    if compiled.search(transcript) or compiled.search(_compact(transcript)):
        return True
    alts = _literal_alts(pattern)
    if alts is None:
        return False
    return any(same_referent(transcript, alt) for alt in alts)


def _answers(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("action") == "ANSWER"]


def _session(event: dict) -> dict:
    session = event.get("session") or {}
    return session if isinstance(session, dict) else {}


def _probes(event: dict) -> dict:
    probes = event.get("probes") or {}
    return probes if isinstance(probes, dict) else {}


def tts_ops(log: str) -> list[dict]:
    """Pause and resume marks, each stamped with the latest [Nms] already seen."""
    last_ms = None
    ops = []
    for line in log.splitlines():
        found = LOG_MS_RE.search(line)
        if found:
            last_ms = int(found.group(1))
        paused = PAUSE_RE.search(line)
        if paused:
            ops.append({"op": "pause", "id": paused.group(1).strip(), "ms": last_ms})
            continue
        resumed = RESUME_RE.search(line)
        if resumed:
            ops.append({"op": "resume", "id": resumed.group(1).strip(), "ms": last_ms})
    return ops


def resume_closes_pause_at_or_after(log: str, t_ms: int) -> bool:
    """True only when a pause at or after t_ms is closed by a later resume of that id.

    A resume whose pause started earlier does not count. That is the historical
    tts.resume ban.
    """
    opened: dict[str, int] = {}
    for op in tts_ops(log):
        ms = op["ms"]
        if ms is None or ms < t_ms:
            if op["op"] == "resume":
                opened.pop(op["id"], None)
            continue
        if op["op"] == "pause":
            opened[op["id"]] = ms
        elif op["op"] == "resume" and op["id"] in opened:
            return True
    return False


def _is_backchannel(text: str) -> bool:
    stripped = text.strip()
    compact = _compact(stripped)
    if not compact:
        return False
    if BACKCHANNEL_RE.match(stripped):
        return True
    if re.fullmatch(r"[嗯恩]+", compact):
        return True
    return compact in {"对", "哦", "好的", "好"}


def _is_interrupt_yield(event: dict) -> bool:
    return event.get("action") == "YIELD"


def _first_answer_ms(events: list[dict], pattern: re.Pattern[str] | None = None) -> int | None:
    for event in _answers(events):
        text = _transcript(event)
        if pattern is not None and not pattern.search(text) and not pattern.search(_compact(text)):
            continue
        return int(event.get("t_ms") or 0)
    return None


def _obs_patterns(clip: dict) -> list[str]:
    return [
        str(obs["pattern"])
        for obs in clip.get("observations") or []
        if obs.get("kind") == "answer_target_regex" and obs.get("pattern")
    ]


_ASK_AGAIN = "抱歉，能请您再说一遍吗？"
_FILLER_RE = re.compile(r"我查一下|正在检索")
_TTS_LINE_RE = re.compile(r"\[tts\.(?:start|end) [^\]]*\]\s*(.*)")
_CLOSING_KEYS = ("不用了", "没事了")


def _event_text(event: dict) -> str:
    return ((event.get("text") or "") + _transcript(event)).strip()


def _utter_matches(event: dict, sop: str) -> bool:
    if event.get("action") != "UTTER":
        return False
    if sop == "01":
        return _ASK_AGAIN in _event_text(event)
    return str(event.get("sop") or "") == sop


def _tts_lines(log: str) -> list[str]:
    lines = []
    for line in log.splitlines():
        found = _TTS_LINE_RE.search(line)
        if found and found.group(1).strip():
            lines.append(found.group(1).strip())
    return lines


def check_utter_sop(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    sop = str(obs.get("sop") or "")
    hits = [int(e.get("t_ms") or 0) for e in events if _utter_matches(e, sop)]
    if hits:
        if sop == "01":
            return True, f"UTTER {_ASK_AGAIN} @ {hits[0]}"
        return True, f"UTTER sop={sop} @ {hits[0]}"
    if sop == "01":
        return False, f"no UTTER of {_ASK_AGAIN}"
    return False, f"no UTTER sop={sop}"


def check_no_answer_before(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    before = str(obs.get("before") or "")
    if not before.startswith("utter_sop:"):
        return False, f"unsupported before={before}"
    sop = before.split(":", 1)[1]
    gate = next((i for i, e in enumerate(events) if _utter_matches(e, sop)), None)
    if gate is None:
        if sop == "01":
            return False, f"no UTTER of {_ASK_AGAIN} to gate"
        return False, f"no UTTER sop={sop} to gate"
    earlier = [
        _transcript(e)[:40]
        for e in events[:gate]
        if e.get("action") == "ANSWER"
    ]
    if earlier:
        return False, f"ANSWER before sop={sop}: {earlier}"
    return True, f"no ANSWER before UTTER sop={sop}"


def check_answer_target_regex(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    pattern = str(obs["pattern"])
    hits = []
    seen = []
    for event in _answers(events):
        text = _transcript(event)
        seen.append(text[:40])
        if transcript_matches_pattern(text, pattern):
            hits.append((event.get("t_ms"), text[:40]))
    if hits:
        return True, f"ANSWER {hits[0]}"
    return False, f"no ANSWER matching /{pattern}/; answers={seen}"


def check_answer_target_contains(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    needles = [str(x) for x in (obs.get("any_of") or [])]
    seen = []
    for event in _answers(events):
        text = _transcript(event)
        seen.append(text[:40])
        if any(same_referent(text, needle) for needle in needles):
            return True, f"ANSWER names {needles} @ {event.get('t_ms')}: {text[:40]}"
    return False, f"no ANSWER naming {needles}; answers={seen}"


def check_no_hard_yield_on_backchannel(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    bad = []
    for index, event in enumerate(events):
        if event.get("action") != "YIELD":
            continue
        if _yield_in_greet(events, index):
            continue
        text = _transcript(event)
        if _is_backchannel(text):
            bad.append((event.get("t_ms"), text))
    if bad:
        return False, f"backchannel YIELD {bad}"
    return True, "no backchannel YIELD"


def _yield_in_greet(events: list[dict], index: int) -> bool:
    """A YIELD still inside the greeting, including the tick that leaves GREETING.

    The scored sample often already reads LISTENING. It still belongs to the
    greeting when no content ANSWER has happened and the session was GREETING
    up to this sample.
    """
    session = events[index].get("session") or {}
    if session.get("mode") == "GREETING":
        return True
    last_mode = None
    for prev in events[:index]:
        if prev.get("action") == "ANSWER":
            return False
        mode = (prev.get("session") or {}).get("mode")
        if mode:
            last_mode = mode
    return last_mode == "GREETING"


def check_no_hard_yield_on_false_barge_script(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    bad = []
    for index, event in enumerate(events):
        if event.get("action") != "YIELD":
            continue
        if _yield_in_greet(events, index):
            continue
        text = _transcript(event)
        if FALSE_BARGE_SCRIPT_RE.search(text):
            bad.append((event.get("t_ms"), text[:40]))
    if bad:
        return False, f"hard YIELD on false barge {bad}"
    return True, "no hard YIELD on false-barge script"


def _target_answer_ms(events: list[dict], clip: dict) -> int | None:
    for obs in clip.get("observations") or []:
        if obs.get("kind") == "answer_target_regex" and obs.get("pattern"):
            for event in _answers(events):
                if transcript_matches_pattern(_transcript(event), str(obs["pattern"])):
                    return int(event.get("t_ms") or 0)
        if obs.get("kind") == "answer_target_contains":
            needles = [str(x) for x in (obs.get("any_of") or [])]
            for event in _answers(events):
                text = _transcript(event)
                if any(same_referent(text, needle) for needle in needles):
                    return int(event.get("t_ms") or 0)
    return _first_answer_ms(events)


def check_false_barge_soft_resume_or_ignore(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    answer_ms = _target_answer_ms(events, clip)
    if answer_ms is None:
        return False, "no target ANSWER before the false-barge window"
    hard = []
    for event in events:
        if event.get("action") != "YIELD":
            continue
        t_ms = int(event.get("t_ms") or 0)
        if t_ms < answer_ms:
            continue
        text = _transcript(event)
        if FALSE_BARGE_SCRIPT_RE.search(text):
            hard.append((t_ms, text[:40]))
    if hard:
        return False, f"hard YIELD on false barge {hard}"

    if resume_closes_pause_at_or_after(log, answer_ms):
        return True, "pause after the answer then tts.resume"

    script_seen = False
    for event in events:
        t_ms = int(event.get("t_ms") or 0)
        if t_ms < answer_ms:
            continue
        if FALSE_BARGE_SCRIPT_RE.search(_transcript(event)):
            script_seen = True
            break
    if script_seen:
        return True, "false-barge script heard after ANSWER and not yielded"
    return False, "no soft resume and false-barge script not heard after ANSWER"


def check_attention_pause_then_resume(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    answer_ms = _first_answer_ms(events, WEATHER_RE)
    if answer_ms is None:
        return False, "no weather ANSWER before the attention window"
    playing_during_away = False
    away_seen = False
    for event in events:
        t_ms = int(event.get("t_ms") or 0)
        if t_ms < answer_ms:
            continue
        session = _session(event)
        vision = event.get("vision") or {}
        attention = str(vision.get("attention") or "")
        away = any(tok in attention for tok in AWAY_TOKENS)
        if away:
            away_seen = True
        if away and session.get("playback_active"):
            playing_during_away = True
    if resume_closes_pause_at_or_after(log, answer_ms):
        return True, "pause after the weather answer then tts.resume"
    if away_seen and not playing_during_away and not any(
        op["op"] == "pause" and op["ms"] is not None and op["ms"] >= answer_ms
        for op in tts_ops(log)
    ):
        return True, "away while TTS already idle; pause not required"
    return False, "no attention soft pause after the weather ANSWER"


def check_no_answer_or_utter_phone_selftalk(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    bad = []
    for event in events:
        if event.get("action") not in {"ANSWER", "UTTER"}:
            continue
        text = _transcript(event)
        if PHONE_RE.search(text) and not WEATHER_RE.search(text):
            bad.append((event.get("t_ms"), event.get("action"), text[:40]))
    if bad:
        return False, f"phone selftalk {bad}"
    return True, "no phone ANSWER/UTTER"


def check_has_second_answer_or_insert(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    answers = _answers(events)
    if len(answers) >= 2:
        return True, f"ANSWER x{len(answers)}"
    if len(answers) == 1:
        first = int(answers[0].get("t_ms") or 0)
        later = [
            e.get("t_ms")
            for e in events
            if e.get("action") == "INSERT" and int(e.get("t_ms") or 0) > first
        ]
        if later:
            return True, f"INSERT after first ANSWER @ {later[0]}"
    return False, f"ANSWER count={len(answers)} and no later INSERT"


def check_answer_count_at_least(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    n = int(obs.get("n") or 0)
    got = len(_answers(events))
    if got >= n:
        return True, f"ANSWER x{got}"
    return False, f"ANSWER x{got} < {n}"


def check_emotion_field_present_on_answer(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    patterns = _obs_patterns(clip)
    targets = []
    for event in _answers(events):
        text = _transcript(event)
        if patterns and not any(transcript_matches_pattern(text, pattern) for pattern in patterns):
            continue
        targets.append(event)
    if not targets:
        return False, "no target ANSWER to read emotion from"
    missing = []
    for event in targets:
        emotion = str((event.get("speech") or {}).get("emotion") or "").strip()
        if emotion:
            return True, f"emotion={emotion} @ {event.get('t_ms')}"
        missing.append((event.get("t_ms"), _transcript(event)[:40]))
    return False, f"target ANSWER has empty emotion {missing}"


def check_no_answer_transcript_regex(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    pattern = re.compile(obs["pattern"])
    bad = []
    for event in _answers(events):
        text = _transcript(event)
        if pattern.search(text) or pattern.search(_compact(text)):
            bad.append((event.get("t_ms"), text[:40]))
    if bad:
        return False, f"ANSWER matched /{obs['pattern']}/ {bad}"
    return True, f"no ANSWER matching /{obs['pattern']}/"


def _repair_sops(clip: dict) -> set[str]:
    for obs in clip.get("observations") or []:
        if obs.get("kind") == "utter_repair" and obs.get("sops"):
            return {str(s) for s in obs["sops"]}
    return set(REPAIR_SOPS)


def check_utter_repair(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    sops = {str(s) for s in (obs.get("sops") or [])}
    hits = [
        (e.get("t_ms"), str(e.get("sop") or ""))
        for e in events
        if e.get("action") == "UTTER" and str(e.get("sop") or "") in sops
    ]
    if hits:
        return True, f"UTTER repair {hits[0]}"
    return False, f"no UTTER in sops {sorted(sops)}"


def check_no_answer_before_repair(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    sops = _repair_sops(clip)
    gate = next(
        (
            i
            for i, e in enumerate(events)
            if e.get("action") == "UTTER" and str(e.get("sop") or "") in sops
        ),
        None,
    )
    if gate is None:
        return False, f"no repair UTTER in {sorted(sops)}"
    earlier = [
        _transcript(e)[:40] for e in events[:gate] if e.get("action") == "ANSWER"
    ]
    if earlier:
        return False, f"ANSWER before repair: {earlier}"
    return True, "no ANSWER before repair UTTER"


def check_has_wait_or_deferred_answer(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    waits = [e.get("t_ms") for e in events if e.get("action") == "WAIT"]
    if waits:
        return True, f"WAIT @ {waits[0]}"
    return False, "no WAIT"


def check_has_yield(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    hits = [
        (e.get("t_ms"), _transcript(e)[:40])
        for index, e in enumerate(events)
        if _is_interrupt_yield(e) and not _yield_in_greet(events, index)
    ]
    if hits:
        return True, f"YIELD {hits[0]}"
    yields = [
        (e.get("t_ms"), _transcript(e)[:20])
        for e in events
        if e.get("action") == "YIELD"
    ]
    return False, f"no YIELD; yields={yields}"


def check_has_answer_after_yield(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    yield_ms = []
    for index, event in enumerate(events):
        if _is_interrupt_yield(event) and not _yield_in_greet(events, index):
            yield_ms.append(int(event.get("t_ms") or 0))
    if not yield_ms:
        return False, "no YIELD before a later ANSWER"
    later = [
        (e.get("t_ms"), _transcript(e)[:40])
        for e in _answers(events)
        if int(e.get("t_ms") or 0) > min(yield_ms)
    ]
    if later:
        return True, f"ANSWER after YIELD {later[0]}"
    return False, "no ANSWER after YIELD"


def check_agent_filler_to_tts(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    """垫话由 Agent 发给 TTS。InteractionCore 自己说出「我查一下」不算。

    垫话不必是「我查一下」这五个字。迎宾和固定开口是 InteractionCore 的，不算。
    """
    ic_lines = {
        (event.get("text") or "").strip()
        for event in events
        if event.get("action") in {"UTTER", "GREET"} and (event.get("text") or "").strip()
    }
    for event in events:
        if event.get("action") == "UTTER" and _FILLER_RE.search(_event_text(event)):
            return False, "InteractionCore uttered the filler"
    for line in _tts_lines(log):
        if line in ic_lines:
            continue
        return True, f"agent filler on TTS: {line[:40]}"
    return False, "agent did not send the filler to TTS"


def check_closing_handed_to_agent(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    """把「不用了」或「没事了」交给 Agent，由 Agent 说结束语。光有 END 不算。"""
    handed = [
        event
        for event in _answers(events)
        if any(key in _compact(_transcript(event)) for key in _CLOSING_KEYS)
    ]
    if not handed:
        return False, "did not hand 不用了 or 没事了 to Agent"
    ic_lines = {
        (event.get("text") or "").strip()
        for event in events
        if event.get("action") in {"UTTER", "GREET"} and (event.get("text") or "").strip()
    }
    for line in _tts_lines(log):
        if line in ic_lines:
            continue
        if _compact(line) in _CLOSING_KEYS:
            continue
        return True, f"Agent closing line on TTS: {line[:40]}"
    return False, "Agent did not speak the closing line"


def check_has_end_or_session_close(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    """离开要交出 END。session_end_pending 不算结束。"""
    ends = [e.get("t_ms") for e in events if e.get("action") == "END"]
    if ends:
        return True, f"END @ {ends[0]}"
    return False, "no END"


def check_has_greet(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    greets = [e.get("t_ms") for e in events if e.get("action") == "GREET"]
    if greets:
        return True, f"GREET @ {greets[0]}"
    return False, "no GREET"


def check_has_greet_or_answer(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    ok, detail = check_has_greet(obs, events, log, clip)
    if ok:
        return ok, detail
    if _answers(events):
        return True, f"ANSWER x{len(_answers(events))}"
    return False, "no GREET and no ANSWER"


def check_no_greet(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    greets = [e.get("t_ms") for e in events if e.get("action") == "GREET"]
    if greets:
        return False, f"GREET @ {greets}"
    return True, "no GREET"


def check_has_end_or_soft_exit(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    ends = [
        (e.get("t_ms"), e.get("sop"))
        for e in events
        if e.get("action") == "END"
    ]
    if ends:
        return True, f"END {ends[0]}"
    return False, "no END"


def check_second_subject_gets_greet(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    end_ms = next(
        (int(e.get("t_ms") or 0) for e in events if e.get("action") == "END"),
        None,
    )
    if end_ms is None:
        return False, "no END before the second GREET"
    later = [
        e.get("t_ms")
        for e in events
        if e.get("action") == "GREET" and int(e.get("t_ms") or 0) > end_ms
    ]
    if later:
        return True, f"GREET @ {later[0]} after END @ {end_ms}"
    return False, f"no GREET after END @ {end_ms}"


def _intro_answer_times(events: list[dict]) -> list[int]:
    times = []
    for event in _answers(events):
        text = _compact(_transcript(event))
        if INTRO_RE.search(text):
            times.append(int(event.get("t_ms") or 0))
    return times


def check_answer_intro_magic_twice(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    times = _intro_answer_times(events)
    if len(times) >= 2:
        return True, f"intro ANSWER x{len(times)} {times}"
    if len(times) == 1:
        return False, f"only one intro ANSWER @ {times[0]}"
    return False, "no intro ANSWER"


def check_answer_intro_magic_once(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    times = _intro_answer_times(events)
    if times:
        return True, f"intro ANSWER @ {times[0]}"
    return False, "no intro ANSWER"


def check_second_intro_barge_then_answer(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    times = _intro_answer_times(events)
    if len(times) < 2:
        return False, "need two intro ANSWERs"
    t0, t1 = times[0], times[1]
    hit = next(
        (
            e.get("t_ms")
            for e in events
            if _is_interrupt_yield(e) and t0 <= int(e.get("t_ms") or 0) <= t1
        ),
        None,
    )
    if hit is None:
        return False, f"no YIELD between intro ANSWERs {t0}..{t1}"
    return True, f"YIELD @ {hit} between intros"


def check_no_answer_non_subject_aside(obs: dict, events: list[dict], log: str, clip: dict) -> tuple[bool, str]:
    bad = []
    for event in _answers(events):
        text = _transcript(event)
        if INTRO_STEM_RE.search(_compact(text)):
            continue
        role = str(_probes(event).get("utterance_role") or "").strip().lower()
        if role == "request":
            continue
        bad.append((event.get("t_ms"), text[:40], role or "missing"))
    if bad:
        return False, f"non-request ANSWER {bad}"
    return True, "no non-request aside ANSWER"


CHECKS = {
    "utter_sop": check_utter_sop,
    "no_answer_before": check_no_answer_before,
    "answer_target_regex": check_answer_target_regex,
    "answer_target_contains": check_answer_target_contains,
    "no_hard_yield_on_backchannel": check_no_hard_yield_on_backchannel,
    "false_barge_soft_resume_or_ignore": check_false_barge_soft_resume_or_ignore,
    "no_hard_yield_on_false_barge_script": check_no_hard_yield_on_false_barge_script,
    "attention_pause_then_resume": check_attention_pause_then_resume,
    "no_answer_or_utter_phone_selftalk": check_no_answer_or_utter_phone_selftalk,
    "has_second_answer_or_insert": check_has_second_answer_or_insert,
    "answer_count_at_least": check_answer_count_at_least,
    "emotion_field_present_on_answer": check_emotion_field_present_on_answer,
    "no_answer_transcript_regex": check_no_answer_transcript_regex,
    "utter_repair": check_utter_repair,
    "no_answer_before_repair": check_no_answer_before_repair,
    "has_wait_or_deferred_answer": check_has_wait_or_deferred_answer,
    "has_yield": check_has_yield,
    "has_answer_after_yield": check_has_answer_after_yield,
    "has_insert_or_processing": check_agent_filler_to_tts,
    "agent_filler_to_tts": check_agent_filler_to_tts,
    "closing_handed_to_agent": check_closing_handed_to_agent,
    "has_end_or_session_close": check_has_end_or_session_close,
    "has_greet": check_has_greet,
    "has_greet_or_answer": check_has_greet_or_answer,
    "no_greet": check_no_greet,
    "has_end_or_soft_exit": check_has_end_or_soft_exit,
    "second_subject_gets_greet": check_second_subject_gets_greet,
    "answer_intro_magic_twice": check_answer_intro_magic_twice,
    "second_intro_barge_then_answer": check_second_intro_barge_then_answer,
    "no_answer_non_subject_aside": check_no_answer_non_subject_aside,
    "answer_intro_magic_once": check_answer_intro_magic_once,
}


def score_clip(clip: dict, events: list[dict], log: str) -> dict:
    results = []
    all_ok = True
    for obs in clip.get("observations") or []:
        kind = obs.get("kind")
        fn = CHECKS.get(kind)
        if fn is None:
            results.append({"kind": kind, "ok": False, "detail": "unknown observation"})
            all_ok = False
            continue
        ok, detail = fn(obs, events, log, clip)
        row = {"kind": kind, "ok": ok, "detail": detail}
        for key, value in obs.items():
            if key != "kind":
                row[key] = value
        results.append(row)
        if not ok:
            all_ok = False
    if not events and not log:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if all_ok else "FAIL"
    return {
        "id": clip.get("id"),
        "suite": clip.get("suite"),
        "status": clip.get("status"),
        "verdict": verdict,
        "ic_gate": clip.get("ic_gate"),
        "agent_gate": "not_scored",
        "agent_gate_text": clip.get("agent_gate"),
        "observations": results,
        "expect": clip.get("expect"),
    }


def score_paths(clip: dict, events_path: Path, log_path: Path) -> dict:
    report = score_clip(clip, load_events(events_path), load_log(log_path))
    report["events"] = str(events_path)
    report["log"] = str(log_path)
    return report
