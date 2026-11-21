# 05 — Validate all runnable Clips

Type: task
Status: resolved
Blocked by: 03

## Scope

After the core change is merged:

1. Run the IC unit tests (`vendor/interactioncore`).
2. Run `supp-9554` and `v2-15` (`15_1_1026.mkv`) replays against the live orchestrator; inspect the IC events for HIGH turns and the expected ANSWERs.
3. Run every `status=runnable` Clip in `test_data/tdd/cases.json` (54: v2 30, v1_new 21, supplement 3) with `run_suite.py`. Blocked Clips stay excluded. `agent_gate` is soft. SOP04 keeps only `04_1026.mkv`.
4. If a Clip fails, classify it: real answer loss / test fixture outdated / environment. Fix real regressions and re-run; do not relax the HIGH gate to make a Clip pass.

## Acceptance

- All 54 runnable Clips PASS (or the failure is reported with evidence and a concrete next step).
- SOP04 has exactly one runnable Clip (`v2-04-1`, `04_1026.mkv`).
- `supp-9554` has 4 ANSWERS, both `交互体验` answers, at least 3 TTS segments, and no <0.05s TTS cancel.
- `v2-15` still ANSWERs and its turn is HIGH.

## Comments

### 2026-11-17 — validation status after the content-hardcode policy

SOP04 is reduced to `v2-04-1` (`04_1026.mkv`); `v2-04-2` and `v2-04-3` were
removed from `cases.json`. The runnable suite is now 54 Clips (v2 30, v1_new
21, supplement 3).

`AGENTS.md` now forbids matching hardcoded ASR/VLM content (only the
backchannel regex is allowed); content and role judgments must use StartLux.
The `你不要再说了` entry added to `_STOP_PHRASES` only to pass `v1-A07-3` has
been reverted (submodule `36242e7`, superproject `b9be969`). This ticket stays
open because the suite is not all-green under that policy.

What was verified before the revert:

- InteractionCore unit tests: 328 passed after the revert.
- Targeted reruns on the final core change: `v2-04-2`, `v2-16`, `supp-279`,
  `supp-9554` PASS. `supp-9554` gives 4 ANSWERs, both `交互体验` utterances,
  6 answer-window TTS segments, no start cancel. `v2-15` ANSWERs at HIGH.
- Full suite (with the now-reverted stop-phrase variant): 53/56 in the first
  run (before SOP04 was reduced). `v2-02-2` (stuck `playback_active`), `v2-10-2` (StartLux role=aside),
  `v2-16` (VLM attention flap) PASSed on individual reruns. `v1-A07-3` only
  passed with the forbidden hardcode.
- The final full-suite run was stopped: the orchestrator was still running the
  forbidden variant, so its results are not valid evidence.

`v1-A07-3` without the hardcode (observed 17:12):

- ASR emits `你不要再说了。`; IC never yields. `v1-A07-3` expects a YIELD.
- StartLux at the same frame: `barge_class=true_interrupt` 0.94
  (`utterance_role=exit` 0.49 on the partial, 0.69 on the punctuated final).
- StartLux is not asked for `barge_class` during playback (commit `f0b35e0`
  disabled it deliberately), and the playback YIELD path is guarded by
  `is_stop_phrase`.
- A policy-compliant fix is therefore a design change, not a phrase:
  ask `barge_class` on a HIGH subject turn during playback and map a committed
  `true_interrupt` (or a committed `exit` role) to a stop YIELD. That changes
  `test_three_way_class_does_not_abandon_playback` and the hard-barge policy
  path, so it needs a decision before implementation.

### 2026-11-18 — final validation on the binary-probe code

Code under test: `vendor/interactioncore` `699ceea` (superproject `f827aec`).
`expects_reply` uses the default-yes wording (`14ece49`/`0c51544`), the content
wait clock starts only when the probe can be asked (`fbce4d5`), and a partial
content verdict is not re-asked for the same sentence (`699ceea`).

Runs (shadow orchestrator on `:8101`, restarted per code version):

| run | code | result |
|---|---|---|
| `/tmp/sop_tdd_final3` | `0c51544` | 51 PASS / 3 FAIL — `v2-10-2`, `v1-A07-8`, `supp-279` |
| `/tmp/sop_tdd_rerun4` | `0c51544` | the three failures re-run: all PASS |
| `/tmp/sop_tdd_final4` | `0c51544` | 53 PASS / 1 FAIL — `supp-279` only |
| `/tmp/sop_tdd_final5` | `699ceea` | **54 PASS / 0 FAIL** (v2 30, v1_new 21, supplement 3) |

Failure classification in `final3`:

- `v2-10-2` `answer_count_at_least`: second line HIGH at t=33400; the content
  verdict `expects_reply=yes 0.636` came back at t=35900 (2.5 s). The pending
  turn was waived at t=34900 (`content-verdict timeout after 1519ms`, shadow log
  22:52:23.976). Rerun: verdict in 504 ms (`yes 0.68`) → PASS.
- `v1-A07-8` `has_answer_after_yield`: second line HIGH at t=29700; no content
  verdict was applied before the clip ended, pending turn waived at t=31200
  (log 23:04:44.899). Rerun: verdict in 305 ms (`yes 0.97`) → PASS.
- `supp-279` `answer_intro_magic_once`: intro HIGH at t=9400–9900 while the VLM
  attention read `侧身或背对`; `attention_away_leftover` consumed the turn. At
  the same time StartLux said `addressed=0.69`, `expects_reply=yes 1.00`,
  `is_closing=no`; attention was back on screen at t=10000. Same clip answers at
  t=9700 when the VLM attention stays on screen (rerun, `final5`).

Classification: StartLux latency tails and VLM attention text, not an
answer-gate or content-probe regression. `supp-279` is the VLM attention debt
recorded in ADR 0002; the candidate follow-up is to let a committed `addressed`
probe outrank the VLM attention string, which is a design decision.

Other checks:

- InteractionCore unit tests: 333 passed, 1 skipped.
- `orchestrator/tests/test_playback_question.py`: 4 passed after the fixture was
  updated to the restored `barge_class` trigger (`1f25e5c`). The three tests had
  asserted `barge_class` is not asked, which ticket 06 deliberately reversed.
- `orchestrator/tests/test_ic_report_reliable.py`: 2 failures reproduce on the
  pre-ticket-06 IC (`36242e7`), so they are pre-existing.
- Code review (integration branch): fixed the partial-content re-ask flood
  (`699ceea`); the other findings are in code from 11/10–11/14 (synchronous
  `_ask_pending_question`, duck recovery, farewell/greet clocks, report throttle)
  and are listed here as pre-existing follow-ups, not changed by this ticket.
- SOP04: exactly one runnable Clip (`v2-04-1`, `04_1026.mkv`).
- `supp-9554`: PASS with 4 ANSWERs, both `交互体验` answers, and the
  answer-window TTS checks. `v2-15`: PASS (ANSWER, HIGH turn).

## Answer

The binary-probe implementation passes the full runnable suite: 54/54 at
`699ceea` (`/tmp/sop_tdd_final5`). Earlier full runs lost single answers to
StartLux latency tails (`v2-10-2`, `v1-A07-8`) and a VLM attention flap
(`supp-279`); all three pass on rerun with the same code, and the final run is
green. InteractionCore tests are 333 passed / 1 skipped. Remaining review
findings are pre-existing code from earlier features on the branch and are
recorded here as follow-ups.
