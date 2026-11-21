# IC ANSWER 必须以 HIGH 话轮为准

**Labels**: ready-for-agent

## Problem Statement

回放 `s-9554c311559e-raw.mkv` 时，IC 在一句话还没说完（完成度 LOW）的时候，因为 StartLux 返回 `repair=answer` 就提前发 ANSWER；同一句正文后来到了 HIGH 又被答了一遍；随后一段新的 TTS 在 0.00s 被 YIELD 掐断。用户实际想要的很简单：

- “给我介绍魔珐科技” → IC ANSWER，Agent 正常回复，TTS 正常播放。
- 第一次“跟你的交互体验是这个” → IC ANSWER，Agent 正常回复，TTS 正常播放。
- 第二次“跟你的交互体验是这个” → IC ANSWER，Agent 正常回复，TTS 正常播放。
- “我今天穿这个开会合适吗” → IC ANSWER，Agent 正常回复；这一问接近视频结尾，replay 时 Agent 回复/TTS 可能收不到，不算错。

用户明确了一条规则：**IC 判 ANSWER，话轮必须是 HIGH；StartLux 的 `repair=answer` 只是意见，最终由 IC 决定。** `repair=burst` 这个判断本身不要求 HIGH，但不能绕开“ANSWER 必须 HIGH”。

当前语音感知里，话轮完成度只有 HIGH 和 LOW，没有 MEDIUM。HIGH 表示“大概率说完了”，LOW 表示“肯定没说完”。现有代码在非 HIGH 时也回答，导致上面的提前回答、重复回答和误 YIELD。

## Solution

从用户视角：

- 我说话时，IC 等到我说完（HIGH）再答，不再抢半句。
- 我说完了，即使 StartLux 的结束判断慢了一点，IC 也不会把这句丢掉。
- 我重复同一句话两次，两次都说完了，就得到两次正常回答。
- 每次回答的 TTS 都正常播放，不会在 0.00s 被掐断。
- 我说“不用了”这类结束句时，仍然先得到结束语，而不是被当成普通问题回答。
- 我在打电话、没对着屏幕或明显在说旁白时，即使说完了也不回答。

实现侧：

- IC 的所有 ANSWER 出口都以 `话轮完成度 == HIGH` 为前提；LOW 只等待。
- StartLux 的 `repair=answer` / `burst` 在 LOW 时仍可作为“听感可用”的意见，用来挡住“请再说一遍”，但不能触发 ANSWER。
- 一个 HIGH 话轮在等 StartLux 的结束判断时，用“待回答话轮”保存转写和回答所需证据；判断回来或 1.5 秒超时后，用保存的证据回答或说结束语。
- 播报中的“交接标记”在句子处理完时清掉；只靠这个标记触发 YIELD 时，当前必须有这句话，转写为空不能 YIELD。
- 新片子 `s-9554c311559e-raw.mkv` 已作为 `supp-9554` 加入 TDD 数据集，验收口径见下。

## User Stories

1. 作为一名用户，我希望 IC 等我说完再回答，这样我不会在说到一半时被打断。
2. 作为一名说到一半停了一下的人，我希望 IC 不要提前回答，因为那句话还没结束。
3. 作为一名已经说完（HIGH）的用户，我希望即使 StartLux 的结束判断慢，IC 也能回答，而不是把这句丢掉。
4. 作为一名说“不用了”的用户，我希望先听到结束语，而不是被当成普通问题回答。
5. 作为一名重复说了同一句话两次的用户，我希望两次说完整后都得到回答，而不是第二次被“同句已答”吞掉。
6. 作为一名在播报中插话的用户，我希望播报先停下，等我说完再把这句话交给 Agent。
7. 作为一名用户，我不希望下一段播报刚起播就 0.00s 被掐断，因为上一句的交接标记还没清。
8. 作为一名打电话或没对着屏幕的用户，我希望即使话轮 HIGH，IC 也不把电话里的自语当成问题回答。
9. 作为一名用户，我希望每次回答的 TTS 正常播放完，而不是刚起播就被取消。
10. 作为一名 Agent，我希望 ANSWER 事件里的 `transcript` 是用户真正说的那句，即使 ASR 快照已经被清空。
11. 作为一名 Agent，我希望 `identity_id` / `display_name` 是问这句话的人，而不是 1.5 秒后站在镜头前的人。
12. 作为一名 Agent，我希望 ANSWER 事件里保留这句话的 `affect`、`language`、`emotion`、噪声字段和 `snr_db`，这样情绪、语种和 SOP15 的确认话术才有依据。
13. 作为一名 Agent，我希望 SOP15 的 `noise_burst_detected` 不丢，这样我才能问“刚刚我没听清，你说想去哪”。
14. 作为一名 Agent，我希望第二次“跟你的交互体验是这个”作为一条独立 ANSWER 到达，而不是被当成第一次的重复。
15. 作为一名测试工程师，我希望 `s-9554c311559e-raw.mkv` 作为 `supp-9554` 进入 supplement Suite，验收四句 ANSWER，以及前三次的 Agent TTS；最后一问的 Agent 回复/TTS 允许缺失。
16. 作为一名测试工程师，我希望 TDD 数据集里所有 runnable 的 Clip 都通过（当前 54 条：v2 30 条、v1_new 21 条、supplement 3 条；SOP04 只保留 `04_1026.mkv`）。
17. 作为一名测试工程师，我希望 3 条 blocked 的 Clip 按 status 排除，而不是被静默跳过。
18. 作为一名测试工程师，我希望 SOP15 在改成 HIGH 闸后仍然通过。
19. 作为一名测试工程师，我希望 SOP05 的打电话、低 addressed 闸在 HIGH 话轮下仍然通过。
20. 作为一名测试工程师，我希望 `supp-9554` 里没有 0.00s 的 TTS 取消。
21. 作为一名开发者，我希望测试只用 HIGH 和 LOW，不再使用生产里不存在的 MEDIUM。
22. 作为一名开发者，我希望“LOW + repair=answer”明确断言 WAIT、不 ANSWER。
23. 作为一名开发者，我希望有一个“结束判断晚到”的待回答话轮测试，证明 ASR 清空后仍能答。
24. 作为一名开发者，我希望有一个“1.5 秒超时”的测试，证明这句话按普通回答发出，并记录超时。
25. 作为一名开发者，我希望有一个“等判断期间来了新内容”的测试，证明旧的待回答话轮被丢掉。
26. 作为一名开发者，我希望有一个“同一句再次定稿 / 短附和”的测试，证明待回答话轮不会被误丢。
27. 作为一名开发者，我希望 ANSWER 动作自己带着回答所需的证据，而不是让 runtime 去读已经被清空的状态。
28. 作为一名开发者，我希望交接标记在“同句已答”或门闩 HOLD 这类路径上也能被清掉。
29. 作为一名开发者，我希望只靠交接标记触发的 YIELD 在转写为空时被禁止。
30. 作为一名开发者，我希望待回答话轮命中新内容或被放弃时，能连续处理下一句，不把旧转写留下来。
31. 作为一名运维，我希望 LOW 时忽略 `repair=answer`、结束判断超时、待回答话轮被丢掉这三种情况都有日志。
32. 作为一名运维，我希望 IC 快照里能看到交接标记、待回答话轮、`answered_content_key` 和 `yielded_content_key`，这样 replay 能直接定位。
33. 作为一名维护者，我希望这次决定写进 ADR，并把“话轮 / 待回答话轮 / 交接标记”补进 `GLOSSARY.md`。
34. 作为一名维护者，我希望这次不改 StartLux 的模型、不改 ASR 快照的生命周期、不改 vision 通道。
35. 作为一名维护者，我希望 SOP15 的 `15_1_1026.mkv` 先 replay 一次，确认真实完成度是 HIGH；如果实际是 LOW，停下来讨论，而不是先放宽。

## Implementation Decisions

### ANSWER 闸

- 所有 ANSWER 出口都以 `话轮完成度 == HIGH` 为前提。现有非 HIGH 的提前回答路径必须改成 WAIT，不再 ANSWER。
- `repair=answer` 和 `repair=burst` 在 LOW 时仍然被采纳为“正文可用”的意见：它挡住“请再说一遍 / 环境太吵”的听感 UTTER，并让 IC 继续等到 HIGH；它不能产生 ANSWER。
- 不给 burst 开非 HIGH 回答的口子。`burst` 只是一个判断，不是 ANSWER 的通行证。
- LOW 定义成“肯定没说完”，永远不答；HIGH 定义成“大概率说完了”，允许小概率误判。生产里没有 MEDIUM，因此不存在“放宽到 MEDIUM”这个选项。

### 测试档位

- TDD 和单测里的“完整句子”用 HIGH，“半句 / 还在说”用 LOW。
- 现有用 MEDIUM 的话轮 fixture 必须去掉或改成 HIGH（场景是完整句子）或 LOW（场景是半句）。
- `LOW + 有正文 + 期望 ANSWER` 的用例按“半句”重写：先断言不答，再给一个 HIGH 定稿，断言只答一次。

### 待回答话轮

- 只在“所有非结束检查都通过、只剩 StartLux 的结束判断”时创建，避免用清空后的状态重跑主体 / addressed / 打电话 / 旁白检查。
- 结束判断回来时：判“结束”就发结束语，判“不是结束”就发 ANSWER。
- 1.5 秒还没回来：按普通 ANSWER 发出，并记录“结束判断超时”。
- 等待期间出现另一句有内容的话（内容 key 不同）或叫停：丢掉待回答话轮，处理新的一句。
- 等待期间只是同一句再次定稿、短附和或噪声：保留待回答话轮。
- 待回答话轮不是“用没说完的话去答”：它保存的是已经到过 HIGH 的话轮。

### 待回答话轮保存的证据

- `transcript`（HIGH 那一刻的正文）。
- `identity_id` / `display_name`（问这句话的人）。
- 主体 track（HIGH 那一刻绑定的主体，不能换成 1.5 秒后画面里的人）。
- `affect`（label + confidence）。
- `language`。
- `emotion`（IC 事件 payload 保留；当前 Agent 协议没有这个字段，Agent 收不到，但保持事件完整）。
- `noise_burst_detected` / `noise_burst_type`。
- `snr_db`。
- 不保存 `vision`。vision 走独立的 `on_vision` 通道，每次收到有内容的 vision 就转发给 Agent；ANSWER 事件里现有的 `vision` 键不被 Agent 消费。

### ANSWER 动作契约

- ANSWER 动作自己携带上面的证据；runtime 用动作里的值构造 Agent 通知，不再从当前状态读取转写或身份。
- `answered_content_key` 用动作里的转写记录；主体绑定用动作里冻结的主体和身份，不用清空后的当前状态。
- 这样可以避免待回答话轮回答时 Agent 收到空 `transcript`，也顺带修掉日志里 `转写=''` 的误导。

### 交接标记

- `hand_off_abandoned_utterance` 表示“我为了把这句话交给 Agent 停掉了播报，欠这句话一个交接”。
- 一个 HIGH 话轮无论最后走到 ANSWER、去重 HOLD、门闩 HOLD 还是被放弃，处理它的那条路径都要负责清掉标记，不能留到下一句。
- 只靠这个标记触发的 YIELD 必须要求当前还有这句话（转写非空、主体语音）；转写为空不能 YIELD。
- 补两个回归测试：同一句第二次定稿；播报开始时标记还在但转写为空。

### 去重

- 只抑制“同一句话还没被新状态打断就又定稿一次”的立即重复。
- 如果两次相同的正文之间，ASR 状态已经变化（出现空帧、别的半句、或新的话轮），第二次是一个新的 HIGH 话轮，必须回答。
- `supp-9554` 的两次“跟你的交互体验是这个”都必须有 ANSWER，用 `answer_target_count` 计数。

### 可观测性

- IC 快照增加：交接标记、待回答话轮（内容 key、等待时长、准备发 ANSWER 还是结束语）、`answered_content_key`、`yielded_content_key`。
- 日志增加：LOW 时忽略 `repair=answer`；结束判断超时；待回答话轮因新内容被丢掉。

### TDD 数据集

- 新增 `supp-9554`（supplement Suite，runnable），视频 `s-9554c311559e-raw.mkv`。
- 观察项：
  - 至少 4 条 ANSWER。
  - “魔珐 / 介绍”有 ANSWER。
  - “交互体验”正好至少 2 条 ANSWER。
  - “穿这个 / 开会 / 合适”有 ANSWER。
  - 至少 3 段 Agent TTS（前三次回答），且没有低于 0.05s 的 TTS 取消；最后一问接近视频结尾，Agent 回复/TTS 可能收不到，不算错。
- 新增两个 Observation 类型：按 pattern 计数的 `answer_target_count`，检查 TTS 段数和起始取消的 `tts_answers_played`。
- 新 Observation 类型要有对应的 scorer 单测。
- SOP15 的单测完成度从 MEDIUM 改成 HIGH；先 replay `15_1_1026.mkv` 确认真实完成度，如果是 LOW，停下来讨论。

### 文档

- 写一条 ADR，记录“ANSWER 必须以 HIGH 话轮为准；StartLux 的 repair 只是意见；HIGH 话轮可以用待回答话轮跨过结束判断的等待”。
- `GLOSSARY.md` 补“话轮”“待回答话轮”“交接标记”，只放定义，不放实现。

### 边界

- 不改 StartLux 的模型、问题或判定契约。
- 不改 ASR 快照的生命周期；用待回答话轮解决跨快照回答。
- 不改 vision 通道，也不把 vision 塞进待回答话轮。
- 不改 Agent 的回复措辞或 TTS 内容。
- 不为了这次改动调整 repair 探针的触发频率；它仍然服务于听感修复。
- 不做 `policy` 的大范围重构；改动限定在 ANSWER 闸、待回答话轮、交接标记和相关测试。

## Testing Decisions

### 测试缝

优先用现有缝，不新开缝：

1. **IC 决策缝**：Engine 的 tick / apply 行为。现有 `test_p05_policy`、`test_playback_barge`、`test_p05_barge` 这一类测试已经从这里驱动，新增用例也走这里。观察 IC 发出的动作和 Agent 收到的事件，不读内部标记。
2. **TDD replay + scorer 缝**：`cases.json` 的 Clip / Observation、`run_suite` 的 replay、`score` 的判定。`supp-9554` 和所有 runnable Clip 都从这里验收。

不需要新的测试缝。

### 好测试的标准

- 只测外部行为：IC 的动作流（ANSWER / WAIT / YIELD / UTTER / END）、Agent 事件 payload、TTS 日志；不测“某个布尔是不是 true”。
- 每个用例只钉一个用户可见结论：答没答、答的是哪句、是不是在说完之后答、TTS 有没有被 0.00s 取消。
- Clip / Observation 的命名用 `GLOSSARY.md` 的词；一条 Observation 只表达一个可机器判定的条件。

### 单测范围

- 更新“非 HIGH 也答”的 7 条现有用例，按新契约改写。
- 新增：
  - LOW + `repair=answer` → WAIT，不 ANSWER。
  - LOW + `repair=burst` → WAIT，不 ANSWER；到 HIGH 再 ANSWER。
  - HIGH + `repair=answer` → ANSWER（正常路径不受影响）。
  - 待回答话轮：结束判断晚到、ASR 已清空 → 仍用冻结的转写 ANSWER。
  - 待回答话轮：1.5 秒超时 → ANSWER + 日志。
  - 待回答话轮：等待期间新内容 / 叫停 → 丢弃。
  - 待回答话轮：同一句再次定稿 / 短附和 → 保留。
  - Agent 事件里的 `transcript`、identity、affect、language、noise、snr 来自冻结证据。
  - 交接标记在“同句已答”和门闩 HOLD 后清掉。
  - 转写为空时，交接标记不能触发 YIELD。
- 先例：`test_p05_policy` 的动作断言风格、`test_playback_barge` 的播报状态风格、`test_score` 的观察项风格。

### TDD 验收

- 跑 `supp-9554`：4 条 ANSWER，两次“交互体验”，至少 3 段 TTS，没有 0.00s 取消；最后一问的 Agent 回复/TTS 允许缺失。
- 跑 `15_1_1026.mkv`：确认 SOP15 的真实完成度是 HIGH，且 ANSWER 仍发生。
- 跑 supplement / v2 / v1_new 全量 runnable。

## Acceptance Criteria

- **TDD 数据集里所有 runnable 的 Clip 必须 PASS。** 当前是 54 条：v2 30 条、v1_new 21 条、supplement 3 条（含 `supp-9554`）。SOP04 只保留 `04_1026.mkv`。
- 3 条 blocked 的 Clip（`v1-A01`、`v1-A02`、`supp-75`）按 status 排除，不算通过，也不算失败；不允许把它们改成 runnable 来绕过。
- `supp-9554` 必须满足：4 条 ANSWER；两次“跟你的交互体验是这个”各一条；至少 3 段 Agent TTS；没有低于 0.05s 的 TTS 取消。最后一问接近视频结尾，Agent 回复/TTS 允许缺失，不算失败。
- SOP15 在 `15_1_1026.mkv` 上必须仍然 ANSWER；其真实话轮完成度必须是 HIGH。如果 replay 出来是 LOW，停止实现并报告，不得自行放宽到 MEDIUM 或给 burst 开口子。
- 所有 IC 单测通过，包括新增的待回答话轮和交接标记用例。
- LOW 时收到 `repair=answer` 不得产生 ANSWER；HIGH 时才答。
- 待回答话轮回答时，Agent 收到的 `transcript`、identity、affect、language、noise、snr 不得为空或错人。
- 转写为空时，交接标记不得触发 YIELD。
- IC 快照能看到交接标记、待回答话轮、`answered_content_key`、`yielded_content_key`。
- ADR 和 `GLOSSARY.md` 已更新。
- 不接受“只跑部分 runnable Clip 就算过”；验收以完整 runnable 集为准。

## Out of Scope

- 修改 StartLux 的模型、问题集或判定逻辑。
- 放宽 ANSWER 到 MEDIUM（生产里没有这个档）；给 burst 开非 HIGH 回答的口子。
- 改 ASR 快照的生命周期或编排层的快照协议。
- 改 vision 通道，或把 vision 放进 ANSWER / 待回答话轮。
- 改 Agent 的回复措辞、TTS 内容或 Agent 协议里新增 `emotion` 字段。
- 处理 3 条 blocked Clip 的阻塞原因（`v1-A01`、`v1-A02`、`supp-75`）。
- 调整 repair 探针的触发频率。
- `policy` 的大范围重构、其他 SOP 的行为调整。

## Further Notes

- 术语：**话轮**是 ASR 判定的一句话单元。HIGH = 大概率说完了，LOW = 肯定没说完；当前没有 MEDIUM。`GLOSSARY.md` 还没有这个词，随这次改动补上。
- `s-9554c311559e-raw.mkv` 现在跑会红在 TTS（31.0s 的 `tts.cancel` 已播 0.00s）；这是预期的红，修复后必须转绿。
- 第二次“跟你的交互体验是这个”是用户真的又说了一遍，是一条新的 HIGH 话轮，不是第一次的重复；去重只能挡住“同一句还没被新状态打断就又定稿”的立即重复。
- 1.5 秒上限来自现有的结束判断等待；待回答话轮是为了让 StartLux 慢或挂掉时不丢 HIGH 话轮。
- 先 replay `15_1_1026.mkv` 再改代码，避免把 SOP15 的真实档位猜错。
- `supp-9554` 的最后一问接近视频结尾；replay 时可能收不到 Agent 回复和 TTS，不算错。IC 仍应有第 4 条 ANSWER；TTS 只要求前三次。
- 验收规则一句话：**所有 runnable 的 Clip 都要 PASS，一条都不能少。**
