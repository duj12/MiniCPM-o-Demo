# 04 — Record the decision in an ADR and the glossary

Type: task
Status: resolved
Blocked by: none

## Scope

- Add an ADR under `docs/adr/` recording: ANSWER requires a HIGH turn; StartLux `repair=answer` is only an opinion; a HIGH turn may cross the end-verdict wait through a pending turn.
- Add/update `GLOSSARY.md` terms: **话轮**, **待回答话轮**, **交接标记**. Definitions only; no implementation details.
- Use the terms exactly as the spec uses them. Do not invent synonyms.

## Pointers

- Spec: `.scratch/ic-answer-high-turn/spec.md` (Implementation Decisions / 文档).
- Existing glossary: `GLOSSARY.md` at the repo root.
- ADR format: `docs/agents/domain.md`, `.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.3.1/skills/engineering/domain-modeling/ADR-FORMAT.md` and `GLOSSARY-FORMAT.md`.

## Comments

- 2026-11-17: Resolved by `71eb9d5 docs: record HIGH-turn ANSWER gate in ADR and glossary`. New `docs/adr/0001-ic-answer-high-turn.md`; `GLOSSARY.md` gained 话轮 / 待回答话轮 / 交接标记. Merged into the integration branch by fast-forward.
