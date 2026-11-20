# 06 — Replace role/exit/stop lexemes with StartLux binary probes

Type: task
Status: resolved
Blocked by: 03

## Why

`utterance_role` and `exit_intent` conflate three different things: whether the
user expects a reply, whether they are stopping the current reply, and whether
they are ending the whole session. `v1-A07-3` showed the failure: StartLux
returned `utterance_role=exit` and `exit_intent=0.707` for `你不要再说了。`, so IC
said `好的，再见。` instead of stopping the playback. The old code hid this with
the `_STOP_PHRASES` transcript table, which `AGENTS.md` now forbids.

Direct StartLux experiments (live state, 2026-11-17) show two binary questions
separate the cases cleanly:

| line | barge_class | expects_reply | is_closing |
|---|---|---|---|
| 正常问题（介绍展厅） | true_interrupt .71 | yes .99 | no 1.00 |
| 叫停（你不要再说了） | true_interrupt .95 | no .78 | no .88 |
| 叫停+提问（别说了，我问你…） | true_interrupt .98 | yes .86 | no .99 |
| 结束（不用了/没事了/再见） | backchannel .29 | no .81 | yes .82–1.00 |
| 第三人称旁白 | backchannel .65 | no .13 | no .76 |
| 附和（嗯嗯，对） | backchannel .95 | no .80 | no .88 |

All stop-phrase variants (`别说了`, `我不想听了`, `你别再说`, `打住`,
`停我不想讨论这个`) are `is_closing=no`; all farewells are `is_closing=yes`.

## Decisions

1. **Stop / continue** is decided by `barge_class` during playback, at LOW.
   Restore the pre-`f0b35e0` trigger: `playback_active` and
   (`user_speaking >= MEDIUM` or turn HIGH). `true_interrupt` at confidence
   ≥0.55 → YIELD; `backchannel` / `false_barge` → keep playing.
2. **Answer / no answer** is decided by `expects_reply` at HIGH.
   `yes` ≥0.55 → ANSWER; `no` ≥0.55 → no answer; below 0.55 → unknown, no answer,
   re-ask when the raw transcript changes.
3. **Farewell / not farewell** is decided by `is_closing` at HIGH.
   `yes` ≥0.60 → END with the farewell; otherwise not closing.
4. `utterance_role` and `exit_intent` leave the decision path and the snapshot.
5. `_STOP_PHRASES`, `_CLOSING_UTTERANCES`, `is_stop_phrase`,
   `is_closing_utterance`, `finished_closing_utterance`, `first_person_subject_followup`
   and the overlapping transcript heuristics (`_phone_selftalk`,
   `_third_person_aside`, `_not_a_barge`, `_still_a_request`) are removed where
   the probes cover them. `_BACKCHANNEL_LINE` stays. `_ATTENTION_AWAY_TOKENS`
   stays only as VLM legacy debt (no StartLux equivalent).
6. Mis-stop recovery: stay stopped (decision Q4 in the grilling round).

## Decision table

| stage | signal | action |
|---|---|---|
| playback + speaking (LOW) | barge_class=true_interrupt ≥0.55 | YIELD now |
| playback + speaking (LOW) | backchannel / false_barge | keep playing |
| HIGH | is_closing=yes ≥0.60 | stop if needed + farewell |
| HIGH | expects_reply=yes ≥0.55 | stop if needed + ANSWER |
| HIGH | expects_reply=no + is_closing=no | no answer; stay stopped if already stopped |
| HIGH | either below threshold | no guess: no answer, no farewell; re-ask on transcript change |

Pending turn: freeze the HIGH evidence, wait up to `exit_verdict_wait_ms`
(1.5s) for the content probes. `is_closing=yes` → farewell;
`expects_reply=yes` → frozen ANSWER; `expects_reply=no` → drop; timeout →
no answer + log.

## Probe wiring

- `decision_client.py`: add `EXPECTS_REPLY_QUESTION`, `IS_CLOSING_QUESTION`;
  remove `UTTERANCE_ROLE_QUESTION`, `EXIT_INTENT_QUESTION`.
- `decision_probe.py`:
  - `barge_class` trigger restored as in decision 1.
  - HIGH subject turn: submit `{"expects_reply": ..., "is_closing": ...}` in one
    request (the service accepts this pair; four-in-one with `addressed` /
    `affect` returns 422).
  - `turn_complete` keeps `{"addressed", "affect"}` (exit_intent removed; this
    pair is accepted; exact order addressed, affect).
- `state.py`: replace `utterance_role*` / `exit_intent*` probe fields and
  `utterance_role_for_key` with `expects_reply*`, `is_closing*`, and a shared
  content key; update `apply_probe`, stale-drop, turn-id carry-over and snapshot.
- `role.py`: `expects_reply(state)` / `is_closing(state)` helpers (stamped,
  same-turn, confidence-gated); remove `UtteranceRole` and role helpers.
- `barge.py`: `_commit_true_interrupt` commits on `barge_class=true_interrupt`
  + subject speech + not attention-away; remove role/exit gates and role waits.
- `playback_barge.py`: stop is `barge_class`; remove the stop/closing tables;
  keep the backchannel regex and the `对着当前会话` yes/no fallback.
- `policy.py`: YIELD on HARD+true_interrupt; ANSWER requires
  `expects_reply=yes`; farewell on `is_closing=yes`; handoff mark resolved by
  `expects_reply`; remove stop/closing/role checks and the removed heuristics.

## Acceptance

- All IC unit tests pass.
- Decision table covered by IC tests: normal question → ANSWER; stop → YIELD,
  no ANSWER; backchannel → no YIELD, no ANSWER; farewell → END; aside → no
  answer; stop+ask → YIELD + ANSWER.
- `v1-A07-3` PASSes without any transcript phrase table; the replay log shows
  `barge_class=true_interrupt` as the stop signal.
- `AGENTS.md` rule holds: no new hardcoded ASR/VLM content; only
  `_BACKCHANNEL_LINE` remains as a lexical shortcut.
- The 54 runnable Clips remain runnable; the full suite is rerun after the
  refactor.

## Answer

Implemented on the integration branch: submodule `b177bb5` (probe swap) and
`19eba39` (review fixes).

- `expects_reply` / `is_closing` replace `utterance_role` / `exit_intent` in
  `Probes`, `apply_probe`, the turn bump, `role.py`, `snapshot.py` and the
  decision path. `UTTERANCE_ROLE_QUESTION` / `EXIT_INTENT_QUESTION` are gone;
  a HIGH subject turn asks `expects_reply` + `is_closing` in one request, and
  `turn_complete` keeps `addressed` + `affect`.
- `barge_class` is asked during playback (speaking or HIGH) and on a stale
  stamp. `true_interrupt` ≥ 0.55 → YIELD; `backchannel` / `false_barge` keep
  playing. `_STOP_PHRASES`, `_CLOSING_UTTERANCES`, `is_stop_phrase`,
  `is_closing_utterance`, the aside/phone/selftalk transcript heuristics and
  the role waits are deleted; only `_BACKCHANNEL_LINE` remains.
- `policy.py`: `is_closing=yes` → farewell; `expects_reply=yes` → ANSWER;
  `expects_reply=no` → HOLD and consume; missing/low → pending turn for
  `exit_verdict_wait_ms` (1.5s), then no answer. `repair=answer` is advisory
  only and never answers by itself.
- Verdicts stay scoped to their sentence: a half-line negative verdict is
  re-asked when the line grows, a pending turn only accepts a verdict for its
  frozen key, a blank ASR frame cannot reuse a stop stamp, and a waived
  sentence is judged again in a new turn.
- Tests: `python -m pytest -q` in `vendor/interactioncore` → 329 passed,
  1 skipped. `tests/test_content_decision.py` covers the decision table
  (normal question, stop, backchannel, farewell, aside, stop+ask, low
  confidence, long utterance, waiver, growth, backchannel during pending).
- Live evidence (shadow orchestrator on port 8101 at `19eba39`): `v1-A07-3`
  PASS. `barge_class=true_interrupt` reaches 0.77 at 20400ms and IC YIELDs
  then; the same line finalizes `expects_reply=False, is_closing=False` and
  IC stays stopped with no farewell and no ANSWER. The intro question is
  ANSWERed on `expects_reply=True 0.97`.
- Known pre-existing debt, deliberately unchanged: `barge.advance_barge` still
  has no production caller (the live path is
  `playback_barge.prepare_playback_voice`), and `_ATTENTION_AWAY_TOKENS`
  remains until the VLM exposes a structured attention field.
