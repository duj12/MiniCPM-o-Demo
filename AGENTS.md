## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Interaction content discipline (InteractionCore)

InteractionCore must not judge the meaning of speech or vision by matching
hardcoded ASR transcripts or VLM text, and must never add such a match to make a
TDD clip pass.

- **The only allowed lexical shortcut is the backchannel heuristic**
  (`_BACKCHANNEL_LINE` / `_is_backchannel`). A backchannel is a short structural
  token, not a semantic judgment.
- **For transcript / VLM meaning, intent, or attention, use StartLux.**
  Ask the probes (`expects_reply`, `is_closing`, `addressed`, `affect`,
  `repair`, `barge_class`) and let IC make the decision from their results.
- **A failing clip is never fixed by a new phrase or label.** Fix the probe /
  decision path, the fixture, or the perception input. Do not add a wording, a
  synonym, or a VLM label as an exception.
- **Existing lexical rules are legacy debt, not precedent.**
  `_ATTENTION_AWAY_TOKENS` (VLM attention text) is the remaining one and should
  be migrated to a structured perception field; do not extend it or add a new
  transcript phrase table.
