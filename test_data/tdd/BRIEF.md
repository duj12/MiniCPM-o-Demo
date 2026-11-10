# SOP TDD brief（给实现 agent）

## Why

用 `test_data/交互策略SOP视频测试用例.md` 里 **V2 视觉**、**v1_new**、**补充测试** 三块，组织可机器判定的测试数据集，支撑 InteractionCore / 编排侧的 TDD。不要用宽松「有 ANSWER 就算过」当唯一闸。

源 md 已按 Demo 列写清 **IC 判什么 / Agent 答什么**。本 brief 与 `cases.json`（version 2）与之对齐。

## Scope

| Suite | 源章节 | Suite 绿线 |
| --- | --- | --- |
| `v2` | `# v2-视觉版本（安卓大屏）` | 全部 `runnable` clip 必须过 |
| `v1_new` | `# v1_new视觉版本（安卓大屏）` | 全部 `runnable` clip 必须过 |
| `supplement` | `# 补充测试`（s-75 / s-09 / s-279） | 只对 `runnable` 计分；当前 s-75 blocked（等主人声提取），`supp-09`+`supp-279` 都须过 |

不包含：旧 `v1` Windows 表、日会记录。源 md 已不再单独列 `v2-纯语音` 验收表；TDD 仍只跑上述三块。

`v2` 绿线包含 SOP05。`v2-05-1`（`05_1_1026.mkv`，侧身背对打电话）和 `v2-05-2`（`05_2_1026.mkv`，正对镜头打电话、时而看屏幕）都是 `runnable`，和其余 v2 片子一起过。IC 硬闸是答天气、打电话或不面向屏幕时 pause、结束电话并面向屏幕时 resume、电话自语不答。Agent 不用停，不单独判红。

权威定义：**md 里的「预期效果 / Demo 预期」产品语义**。问题描述只作诊断上下文。见仓库根 `GLOSSARY.md` 的「通过」「行为契约」。

## Artifacts

- 本 brief：意图、禁区、IC/Agent 分层
- `cases.json`：每条 Clip 的 id / suite / video / status / expect / observations / `ic_gate` / `agent_gate`
- 源 md：`../交互策略SOP视频测试用例.md`
- 视频：默认相对 `test_data/`

## IC gate vs Agent gate

md Demo 大量写成「ic判answer，agent结合vlm/情绪回复」。

| 层 | 含义 | TDD |
| --- | --- | --- |
| **IC 硬闸** | ANSWER / YIELD / WAIT / UTTER / GREET / END / INSERT、pause/resume、不对旁白 ANSWER | 必须进 `observations`，失败即 Clip 红 |
| **Agent 软备注** | VLM/情绪/多语种措辞、部分理解追问话术 | 写在 `agent_gate` / notes；默认不单独判红，除非 observation 显式要求 |

例：SOP15 突发噪声 → IC 对残缺行程句 **ANSWER**；「没听清去哪」由 **Agent** 追问，不是 IC sop01 硬闸。

## Two test layers

1. **契约层（慢，真）**  
   对 `status=runnable` 的 Clip 做视频回放（`orchestrator_replay` + IC events + 必要时 replay 日志）。断言写在 `observations` 里。

2. **实现层（快，窄）**  
   InteractionCore 单测用合成 state / 探针票，覆盖同一 Observation 的决策分支。单测绿不能单独宣称 Clip 通过。

## Observation truth

允许的证据：

- IC `events.jsonl`：`action`、`speech.transcript`、`session.barge_phase` / `barge_pause_reason`、`probes.utterance_role`、`sop`
- replay 日志：`tts.pause` / `tts.resume`（SOP04 / SOP05）

### ANSWER 转写：语义一致即可

IC 判 `ANSWER` 后交给 Agent 的 `speech.transcript`，**不需要**与 md / `expect` / observation 针里的验收文字逐字相同，**语义（同指称）一致即可**。

- Scorer 用指称/近邻匹配（见 `score.py` 的 `same_referent` / `transcript_matches_pattern`），禁止要求整句 exact match Demo 脚本。
- 仍须钉住**目标话轮**（订会议室、问 AIGC、介绍魔珐等）；无关转写（「哼哼」、闲聊）不能冒充目标 ANSWER。

## Forbidden sole gates（假绿禁区）

下列条件**不能单独**判 PASS：

- 任意时刻出现过一次 `ANSWER`
- 日志里任意历史 `tts.resume`（与本次 soft 无关）
- 问候窗里的 `YIELD` 充当误打断恢复，或当成 SOP03/SOP04「其他处不 YIELD」的失败（greet 过程中的 YIELD 除外）
- 对非目标转写 ANSWER（例如「哼哼」）却算日程/情绪 SOP 通过
- 因 ASR/措辞与 md 验收句不完全一致就判红（语义已对齐时）

## v1_new / A7（8 条视频）

md 写作 `A7 YIELD`。清单 **稳定 id** 仍为 `v1-A07` … `v1-A07-8`（共 **8** 条，勿合并）。

| id | 视频 | 要点 |
| --- | --- | --- |
| `v1-A07` | `A07.mkv` | 「别说了」真打断 |
| `v1-A07-2` … `5` | `A07_2`…`5_1109.mkv` | 直接打断话术 |
| `v1-A07-6` | `A07_6_1109.mkv` | 先 ANSWER 介绍；「想听星云」真打断后再交 Agent |
| `v1-A07-7` | `A07_7_1109.mkv` | 先 ANSWER；「生成式AI」处 YIELD 再交 Agent |
| `v1-A07-8` | `A07_8_1109.mkv` | 先 ANSWER；「魔珐有言」处 YIELD 再交 Agent |

## Supplement notes

- `supp-75` 与 `supp-09` 共享「行为契约」：两次介绍都答；第二次先打断再答；非主体旁白不答
- `supp-279`：主体介绍答一次，不被旁人带走
- **`supp-75` 暂时 `blocked`**（`waiting_subject_voice_extraction`）：md 备注「暂时先不测。等主人声提取ready」。当前只跑 `supp-09` / `supp-279`，二者都须过

## Blocked clips

`status=blocked` 的行要保留在清单里（契约仍在），但不要进可跑套件。当前常见原因：`never_recorded`（v1 A01/A02，本身未录制）；`waiting_subject_voice_extraction`（`supp-75`）。

`v2-06-2` / `v2-06-3`（`06_2_1109.mkv` / `06_3_1109.mkv`）已补片，现为 `runnable`。

`v2-08-2`（`8_2_1026.mkv`）的 Demo 预期与 `v2-08-1` 同句：天气加毛衣问句，IC 判 ANSWER，Agent 结合 VLM。场景列里的「持续远离 / 这个距离」不是这条片子的闸。

md 备注里的 `D21_bug.mkv` **不存在**：不进清单、不测、不算 blocked 契约。

## Suggested agent workflow

1. 读本 brief + `cases.json` + `GLOSSARY.md`
2. 为每个 `observations[].kind` 实现/收紧 scorer（禁止假绿禁区；尊重 IC vs Agent 分层）
3. 先写会红的单测或 scorer 用例，再改 IC / replay
4. 对变更相关的 runnable clips 做视频回放验收（**A7 八条都要覆盖到**）
5. 报告：suite 计数、失败 clip 的 observation 明细、blocked 列表

## Out of scope for the TDD agent

- 改源 md 宽表「预期」措辞（除非操作者明确要求）
- 把 tip / 旁白词表硬编码进 IC
- 未受权就 commit / 开 PR
- 用 viz「听感过关」代替 Observation
- 把 Agent 软备注当成 IC 硬闸（或反过来用任意 ANSWER 冒充 IC 闸）
