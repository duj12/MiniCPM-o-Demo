# cases.json schema（version 2）

## Top-level

| Field | Meaning |
| --- | --- |
| `version` | Manifest schema version（当前 2） |
| `source_md` | 源 md 相对仓库路径 |
| `synced_at` | 上次同步时间（含时区，如 `2026-11-09T13:04:00+08:00`） |
| `suites` | suite 元数据：`pass_rule`、`description`、可选 `notes` |
| `forbidden_sole_gates` | 假绿禁区字符串列表 |
| `observation_truth` | 允许的证据源 |
| `ic_vs_agent` | IC 硬 / Agent 软 的一句话约定 |
| `cases` | Clip 数组 |

## Clip fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | 稳定 id，如 `v2-06-2`、`v1-A07-8`、`supp-09` |
| `suite` | yes | `v2` \| `v1_new` \| `supplement` |
| `sop` | yes | 短标签，如 `06 GESTURE`、`A07 YIELD` |
| `video` | yes* | 文件名或相对 `test_data/`；未录制可为 `null` |
| `status` | yes | `runnable` \| `blocked` |
| `expect` | yes | 压缩后的产品预期 |
| `observations` | yes | 数组；blocked 可 `[]` |
| `source` | yes | `md#...` 回溯 |
| `ic_gate` | yes* | IC 硬闸摘要（blocked 契约行也建议写） |
| `agent_gate` | no | Agent 软备注 |
| `blocked_reason` | blocked 时 | `missing_video` \| `never_recorded` \| `waiting_*` … |
| `notes` | no | 诊断/范围说明 |

## Observation `kind`（常用）

实现 scorer 时以清单为准；生产数据时优先复用已有 kind，勿无故发明同义名。

| kind | 用途 |
| --- | --- |
| `utter_sop` | 出现指定 sop 的 UTTER（如 `"01"`） |
| `no_answer_before` / `no_guess_answer_before_repeat` | 修复话术前不许猜答 |
| `answer_target_regex` / `answer_target_contains` | 对目标话轮 ANSWER；匹配为语义/同指称（`same_referent`），非整句 exact |
| `answer_target_count` | 对同一目标 pattern 的 ANSWER 次数下界（同一句重复出现要分别答） |
| `answer_count_at_least` | ANSWER 次数下界 |
| `tts_answers_played` | 首个 IC ANSWER 之后的 Agent TTS 段数下界；不计迎宾/垫话，`tts.cancel 已播 0.00s` 判红 |
| `has_yield` / `has_answer_after_yield` | 真打断与之后交权 |
| `no_hard_yield_on_backchannel` | 嗯啊等不应硬 YIELD |
| `false_barge_soft_resume_or_ignore` / `no_hard_yield_on_false_barge_script` | 误打断 soft 路径 |
| `attention_pause_then_resume` | SOP04/05 pause→resume |
| `has_wait_or_deferred_answer` | WAIT / 不抢答 |
| `has_greet` / `no_greet` / `has_greet_or_answer` | 招呼策略 |
| `no_answer_non_subject_aside` | 非主体旁白不答 |
| `answer_intro_magic_once` / `answer_intro_magic_twice` | 介绍魔珐次数 |
| `second_intro_barge_then_answer` | 第二次介绍先打断再答 |

字段随 kind 变化：常见 `pattern`、`sop`、`before`。

## Id 命名习惯

- v2：`v2-{sop}-{variant}`，如 `v2-06-2`、`v2-01-1`
- v1_new：`v1-{SOP}`，多视频加后缀 `v1-A07-2`
- supplement：`supp-{短号}`，如 `supp-75`、`supp-09`、`supp-279`、`supp-9554`

## 与实现 agent 的分界

| 角色 | 入口 |
| --- | --- |
| 生产/同步 TDD 数据 | 本 skill |
| 按 TDD 改 IC / replay / scorer | `test_data/tdd/BRIEF.md`（本 skill 只保证 brief 与 json 一致） |
