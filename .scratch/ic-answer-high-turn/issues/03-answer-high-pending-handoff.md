# 03 — IC ANSWER is HIGH-only; pending turn and handoff mark

Type: task
Status: resolved
Blocked by: 01, 02

## Scope

Implement the spec's core decisions in `vendor/interactioncore`:

1. **ANSWER gate** — every ANSWER exit requires `turn_complete_confidence == HIGH`. LOW/MEDIUM must not ANSWER. `repair=answer` / `repair=burst` remain advisory: they may block "ask again" and keep the turn alive, but they cannot trigger ANSWER. Remove the no-MEDIUM relaxation and the burst exception. Rewrite tests that use MEDIUM as a turn-completion fixture; LOW + text + expected ANSWER becomes "WAIT first, then one ANSWER at the HIGH final".
2. **Pending turn** — when all non-end checks have passed and only the StartLux end verdict remains, freeze the HIGH turn's evidence with a 1.5s cap. On verdict END → farewell; on not-END → ANSWER from the frozen evidence; on timeout → ANSWER + log. New meaningful content or a stop phrase drops it; the same content re-final / backchannel keeps it. It does not freeze vision.
3. **ANSWER action contract** — the ANSWER `Action` carries transcript, identity, display name, subject track, affect, language, emotion, noise-burst fields and `snr_db`. `runtime._notify_sinks` uses those values for ANSWER instead of reading the possibly-cleared state. `apply_action` uses the action transcript for `answered_content_key` and the action's subject for the binding. Keep vision on its separate `on_vision` channel.
4. **Handoff mark** — clear `hand_off_abandoned_utterance` when the sentence is resolved (ANSWER / dedupe HOLD / gate HOLD / abandoned). A mark-only YIELD requires a current non-empty subject transcript; an empty transcript must not YIELD. Add the two regression tests.
5. **Dedupe** — suppress only an immediate re-final of the same content key in the same HIGH turn. If the ASR state changed between two identical texts (empty frame, other partial, new turn), the second is a new HIGH turn and must answer. `supp-9554`'s two `交互体验` answers depend on this.
6. **Observability** — snapshot: handoff mark, pending turn (content key, waited ms, target answer/farewell), `answered_content_key`, `yielded_content_key`. Log: LOW ignoring `repair=answer`; end-verdict timeout; pending turn dropped for new content.

## Pointers

- Spec: `.scratch/ic-answer-high-turn/spec.md` (Implementation Decisions, Testing Decisions, Acceptance Criteria).
- Code: `interaction/policy.py` (`_consider_repair` early-answer branch, `exit_answer_gate`, `apply_action`, `decide` handoff/YIELD/dedupe/normal ANSWER), `interaction/state.py`, `interaction/runtime.py` (`_answer_extras`, `_notify_sinks`), `interaction/snapshot.py`, `interaction/decision_probe.py`.
- Tests: `tests/test_p05_policy.py`, `tests/test_policy_sops.py`, `tests/test_slow_release_fast_filler.py`, `tests/test_answer_or_repair.py`, `tests/test_closing_line.py`, `tests/test_p05_barge.py`, `tests/test_playback_barge.py`.

## Acceptance

- All IC unit tests pass.
- `supp-9554`'s two `交互体验` utterances each produce an ANSWER; a LOW partial does not.
- `15_1_1026.mkv` still ANSWERs at HIGH.
- Agent receives the frozen transcript/identity/affect/language/noise/snr for a pending-turn answer.
- The handoff mark cannot produce a YIELD with an empty transcript.

## Answer

Implemented in `vendor/interactioncore` on `ic-answer-high-turn`, merged into the
integration branch as `60f8de2` (plus the review fixes in `29ad8f9` and the
subject-anchor fix `9968673` / `6e4ce04`).

- **ANSWER gate**: every ANSWER exit requires `turn_complete_confidence == HIGH`;
  `repair=answer/burst` only blocks ask-again and keeps the turn alive. LOW logs
  `ignoring repair=answer at turn=LOW` and WAITs.
- **Pending turn**: `exit_answer_gate` freezes the HIGH turn for
  `exit_verdict_wait_ms` (1.5s); END → farewell, not-END → frozen ANSWER,
  timeout → frozen ANSWER + log. New content / stop phrase drops it; same-sentence
  re-final and backchannels keep it. Vision is not frozen.
- **Action contract**: ANSWER carries frozen transcript / identity / display name /
  subject track / affect / language / emotion / noise fields / `snr_db`;
  `_notify_sinks` and `apply_action` use the Action for ANSWER, `on_vision` stays
  separate.
- **Handoff mark**: cleared on ANSWER / dedupe HOLD / gate HOLD / ASIDE / OTHER;
  a mark-only YIELD waits one `post_answer_role_wait_ms` for a missing role and
  requires a non-empty subject transcript.
- **Dedupe**: `answered_turn_id == turn_id` + same content key, with prefix/suffix
  extension suppressed only inside the same HIGH turn.
- **Observability**: snapshot carries `handoff`, `pending_turn` (key, waited ms,
  target answer/farewell), `answered_content_key`, `yielded_content_key`; the
  orchestrator forwards the `turn` block through `IcDisplay` / `_on_ic_action`.
- **Subject anchor** (found while validating `supp-279`): ASR blanks the transcript
  for a beat before the final result; the anchor is now kept for a 1500ms
  finalization window instead of being dropped on the blank frame.

Verification: `vendor/interactioncore` 328 passed; `supp-9554` has 4 ANSWERS with
both `交互体验` utterances and 6 answer-window TTS segments / no start cancel;
`v2-15` answers at HIGH. See ticket 05 for the full Clip run.
