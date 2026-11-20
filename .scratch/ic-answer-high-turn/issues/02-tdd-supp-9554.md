# 02 — Add supp-9554 to the TDD dataset

Type: task
Status: resolved
Blocked by: none

## Scope

Add `test_data/s-9554c311559e-raw.mkv` as `supp-9554` in `test_data/tdd/cases.json` (supplement Suite, runnable), with observations:

- `answer_count_at_least: 4`
- `answer_target_regex: 魔珐|介绍`
- `answer_target_count: 交互体验, n: 2`
- `answer_target_regex: 穿这个|开会|合适`
- `tts_answers_played: n: 3` (the last question is near the video end; its Agent reply/TTS may be missing and is not an error)

Add the `answer_target_count` and `tts_answers_played` Observation checks in `test_data/tdd/score.py`, plus unit tests in `test_data/tdd/test_score.py`.

## Result

Committed on the integration branch as `8ef3ff5 test(tdd): add supp-9554 answer/TTS regression`.

`pytest test_score.py`: 18 passed.

## Comments

- 2026-11-17: The last IC ANSWER is still required; only the last Agent reply/TTS is allowed to be absent. A TTS segment that actually starts and is cancelled below 0.05s still fails.
