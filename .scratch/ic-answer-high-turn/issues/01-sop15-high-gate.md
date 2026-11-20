# 01 — Confirm SOP15's real turn completion is HIGH

Type: research
Status: resolved
Blocked by: none

## Question

Before changing the ANSWER gate, confirm that `test_data/15_1_1026.mkv` really reaches `turn_complete_confidence == HIGH`. If it is LOW, the spec says stop and report; do not relax the gate.

## Answer

The live replay reached HIGH at **11.7s** and the IC emitted ANSWER at **11.9s** for the final transcript `我今天想去北京，给我介绍一下行程吧。`.

Command (orchestrator already on `127.0.0.1:8100`):

```bash
/root/miniconda3/envs/orch105/bin/python -u orchestrator_replay.py \
  --video test_data/15_1_1026.mkv \
  --host 127.0.0.1 --port 8100 \
  --ic-events-out /tmp/sop15_gate/15_1.events.jsonl \
  --no-show --mute --face-size 1280x720 \
  --drain-s 40 --wait-reply-s 20
```

Evidence: `15_1.events.jsonl`, events 72 (`HIGH`) and 73 (`ANSWER`).

The gate passes: SOP15 can be migrated to HIGH without relaxing the spec.

## Comments

- 2026-11-17: Replay exit was 1 because the replay client saw a connection close during the final TTS; the ANSWER action and HIGH turn are present and are what this ticket verifies.
