---
name: sop-tdd-data
description: >-
  Sync InteractionCore SOP video demos into machine TDD artifacts
  (test_data/tdd/cases.json + BRIEF.md) from 交互策略SOP视频测试用例.md.
  Use when producing, updating, or auditing SOP TDD clips, IC/Agent gates,
  blocked reasons, video presence, or A7 multi-video suites.
---

# SOP → TDD 测试数据

把 `test_data/交互策略SOP视频测试用例.md` 的 Demo 预期，同步成可机器判定的 TDD 数据集。本 skill **只生产/维护测试数据与 brief**，不负责改 IC 实现（实现 agent 读 `BRIEF.md`）。

## Artifacts（权威顺序）

| 角色 | 路径 |
| --- | --- |
| 产品语义源 | `test_data/交互策略SOP视频测试用例.md` |
| 机器清单 | `test_data/tdd/cases.json`（`version` ≥ 2） |
| 实现交接 brief | `test_data/tdd/BRIEF.md` |
| 术语 | 根目录 `GLOSSARY.md`（通过 / 行为契约 / Clip / Observation） |
| 视频根 | `test_data/`（`video` 字段相对此目录） |

改清单时：**md 语义为准 → 更新 `cases.json` → 同步 `BRIEF.md` 禁区/blocked/特例**。不要让 brief 与 json 分叉。

## Scope

只收录三块 suite：`v2`、`v1_new`、`supplement`。

不收录：旧 v1 Windows 表、日会记录、已从表中拿掉的纯语音验收表。

Suite 绿线：该 suite 下全部 `status=runnable` 必须过；`blocked` 保留契约但不计分。

## Workflow

复制进度：

```
- [ ] 1. 读源 md 目标章节 + 现有 cases.json / BRIEF.md / GLOSSARY.md
- [ ] 2. 逐行映射为 Clip（稳定 id；一视频一 id）
- [ ] 3. 拆 IC 硬闸 vs Agent 软备注
- [ ] 4. 写 observations（对准目标转写/探针，禁止假绿）
- [ ] 5. 对照 test_data/ 核对视频；设 runnable / blocked
- [ ] 6. 含糊边界先问操作者，再落盘
- [ ] 7. 更新 synced_at；BRIEF 写清特例（A7 条数、supp-75、未录制项）
- [ ] 8. 汇报：runnable 数、blocked 表、相对上次的变更
```

### 1. Clip 映射规则

- **一视频一条** Clip。同一 SOP 多视频 → 多 id，勿合并（例：A7 = `v1-A07` … `v1-A07-8` 共 8 条）。
- `id` 稳定：后续 scorer/报告按 id 对齐；改视频文件名可改 `video`，尽量不改 id。
- `expect` 压缩 md「预期效果 / Demo 预期」；问题描述只作诊断，不单独成闸。
- `source` 指向 md 锚点，便于回溯。
- md **备注里的诊断片**（如不存在的 `D21_bug.mkv`）：**不进清单**，不要伪造 blocked 契约。

### 2. IC gate vs Agent gate

md 常见写法：「ic判answer，agent结合vlm/情绪回复」。

| 层 | 写哪里 | TDD |
| --- | --- | --- |
| IC 硬闸 | `ic_gate` + `observations` | 失败即 Clip 红 |
| Agent 软备注 | `agent_gate` / `notes` | 默认不单独判红，除非 observation 显式要求 |

IC 硬闸典型：`ANSWER` / `YIELD` / `WAIT` / `UTTER` / `GREET` / `END` / `INSERT`、`tts.pause`/`resume`、不对旁白/非主体 ANSWER。

Agent 软备注典型：VLM/情绪/多语种措辞、部分理解追问话术。

**易混例（已拍板可作默认）**：

- SOP15 突发噪声：IC 对残缺行程句 **ANSWER**；「没听清去哪」属 **Agent** 追问，不是 IC sop01。
- SOP02 部分理解：IC **ANSWER** 交 Agent；追问话术是 Agent 软备注。

边界不清时：**先问操作者**，不要猜成更严的 IC 闸。

### 3. Observations（防假绿）

允许证据：

- IC `events.jsonl`：`action`、`speech.transcript`、`barge_phase` / `barge_pause_reason`、`probes.barge_class` / `probes.expects_reply`、`sop`
- replay 日志：`tts.pause` / `tts.resume`（SOP04/05）

**禁止单独作 PASS**：

- 任意时刻出现过一次 `ANSWER`
- 与本次 soft 无关的历史 `tts.resume`
- 问候窗 `YIELD` 冒充误打断恢复
- 对非目标转写 ANSWER（如「哼哼」）却算日程/情绪 SOP 过

Observation 必须钉住**目标话轮或探针**（`answer_target_regex` / `utter_sop` / `has_yield` 等）。schema 与常用 `kind` 见 [reference.md](reference.md)。

**ANSWER 转写**：交给 Agent 的 transcript 与 md/`expect`/needle **语义（同指称）一致即可**，不要求与验收文字逐字相同。写 needle 时用关键指称（「会议室」「AIGC」「魔珐」），由 scorer 做近邻/同指匹配；不要把整句 Demo 脚本当 exact-match 闸。无关转写仍失败。

### 4. 视频与 status

对每条 Clip 在 `test_data/` 下按**文件名**查找：

| 情况 | `status` | `blocked_reason` |
| --- | --- | --- |
| 文件在盘且要测 | `runnable` | （删除该字段） |
| 应测但片尚未到位 | `blocked` | `missing_video` |
| md 有 SOP、**从未录制** | `blocked` | `never_recorded`（`video` 可 `null`） |
| 片在但前置能力未好 | `blocked` | 具体原因（如 `waiting_subject_voice_extraction`） |
| md 备注鬼片/不存在的诊断文件 | **删除条目** | — |

补片后：把对应 Clip 改为 `runnable`，去掉 `blocked_reason`，并改 BRIEF。

补录后自检：

```text
runnable 条数；每条 video 文件名是否都能在 test_data/ 找到；blocked 列表与原因
```

### 5. Supplement / 共享行为契约

同一「预期结果」措辞的 Clip 共享**行为契约**（见 GLOSSARY），不按问题描述拆成两套闸。

当前约定：`supp-75` 与 `supp-09` 同契约；`supp-75` 在主人声提取 ready 前保持 `blocked`；只对 runnable（如 `supp-09` + `supp-279`）计分。

### 6. BRIEF 必须同步的内容

- Suite 绿线与 scope
- IC vs Agent 分层表
- 假绿禁区
- 多视频特例（A7=8）
- blocked 原因与「不进清单」项
- 给实现 agent 的 workflow / out of scope

## Out of scope

- 擅自改宽/改严源 md「预期」措辞（除非操作者明确要求）
- 把 tip / 旁白词表硬编码进 IC
- 未授权 commit / 开 PR
- 用听感/viz 代替 Observation
- 把本 skill 当成「按 TDD 改 IC」的实现流程（那是读 BRIEF 的另一类任务）

## 交付汇报模板

```text
synced_at: ...
runnable: N（磁盘缺失: 0）
blocked:
  - id — reason
变更要点:
  - ...
待裁决（如有）:
  - ...
```
