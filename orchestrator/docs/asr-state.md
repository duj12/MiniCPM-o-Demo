# ASR 五个状态量 —— 取值约定（**权威**）

本文是 `AsrState` 五个字段的**唯一权威定义**。其它位置（`README.md`、
`asr/client.py` 的 docstring、`tests/README.md`）只做摘要 + 链接到本文，
避免多处各写一份、然后各自漂移。

代码：`orchestrator/asr/client.py` 的 `AsrStateTracker`。
下发：`AsrDisplay.state`（UI）/ `apply_vad` `apply_asr`（InteractionCore）。

---

## 一、核心区分：**本轮内** vs **本轮结论**

这五个字段分成两类，**清空时机完全不同** —— 这是本设计最要紧的一条：

| 类别 | 字段 | 回答的问题 | 一轮结束后 |
|---|---|---|---|
| **本轮内** | `说` `抢` | 「**此刻**在不在出声 / 抢话」 | **立即** `NONE` |
| **本轮结论** | `完` `信` `text` | 「**这一轮**说完了吗 / 置信度 / 说了什么」 | **保留 1s**（可调），供 UI 读刚才那句 |

**为什么这么分**：「此刻在不在」是**瞬时**事实 —— 这一轮结束了，答案就作废，
继续报 `HIGH` 是错的。「本轮是什么」是**已经发生**的事实 —— 轮结束后还要让人
看到（UI 上得能读刚才说了什么）。

> 早先两者共用同一个保持窗口（`SPEAKING_HOLD_MS = 1200`），表现为
> 「一轮结束了还在报用户在说话」。已废弃。

### 「一轮结束」的定义

**段关闭**，两处触发：

- 收到 `2pass-offline`（服务端对这段的最终识别）
- 收到 `turnsense`（VAD 判出语音段边界）

### ⚠️ 一拍缓冲：保证「拿到最终结果 = 抢 HIGH」能被观测到

`offline` **同时**是「本轮结束」和「确认说出了内容」—— 两者在同一条消息上。
若在 `offline` 里直接关段，那个 `HIGH` 就**永远观测不到**（快照与关段同拍完成）。

所以 `offline` 那一拍：

```
offline 到达 → 说=HIGH 抢=HIGH（IC 必然收到）    ← 一拍缓冲
下一拍       → 说=NONE 抢=NONE（立即清）
```

实现见 `_pending_close` / `_turn_active()`。**这一拍不是"保持窗口"**，
而是「让同一拍产生的两个事实都被观测到」—— 只有一个 tick（50ms）。

---

## 二、四个档位量

取值域统一 `HIGH` / `MEDIUM` / `LOW` / `NONE`。

### 说 `user_speaking_confidence` —— 用户是否正在出声

| 值 | 条件 |
|---|---|
| `HIGH` | 段开着（含拍板那一拍）**且**本段已有转写文本 |
| `MEDIUM` | 段开着**但还没有文本**（盲窗：出声了、ASR 还没吐字） |
| `NONE` | 段已关且过了缓冲那一拍 = 真静音 |

**用途**：IC 的 `user_is_speaking`（→ `LISTEN`）与 SOP 06。
注意 IC 要求的是 `== HIGH`，`MEDIUM` 不满足。

### 抢 `barge_in_confidence` —— 抢话把握

| 值 | 条件 |
|---|---|
| `HIGH` | **拿到带转写的最终结果**，或本段已吐 **≥3 个**带文本流式帧 |
| `MEDIUM` | 本段 **2 个**带文本流式帧 |
| `LOW` | 本段 **1 个**带文本流式帧 |
| `NONE` | 本段还没有，或段已关 |

**用途**：IC 的 SOP 07 抢话停播 —— `barge_yield_signal` 判据是
`barge≥MEDIUM ∧ face HIGH ∧ lip≥MEDIUM`，且需**连续满足 300ms**（`barge_yield_ms`）
才 `YIELD`。

**两条达到 `HIGH` 的路径**，为什么：

- 「拿到带转写的最终结果」= 用户**确实说出了内容**，最硬的证据
- 「≥3 个带文本流式帧」= ASR 反复确认有内容（应对最终结果来得晚的情况）

⚠️ **空文本的最终结果不置 `HIGH`** —— 那只是 VAD 收尾帧，不代表说出了内容
（与「空帧不擦转写」同一口径，见 `downstream.py` 里那段注释）。

⚠️ **帧数每段重置**（在 `_on_online` 的新段分支里）。早先只在 `offline` 归零，
而 `offline` 不再归零之后，靠的是段开始重置 —— 这样**不跨句累积**
（早先的 bug：句1+1、句2+1、句3 刚开始就 `HIGH`）。

### 信 `asr_confidence` —— 本轮转写置信度

| 值 | 条件 |
|---|---|
| `HIGH` | 平均置信度 `>= confidence_threshold`（默认 **0.8**） |
| `MEDIUM` | `>= online_confidence_threshold`（默认 **0.6**）且 `< 0.8` |
| `LOW` | `< 0.6`，**或**流式文本被服务端过滤掉（空文本帧但带低分） |
| `NONE` | 无置信度信号 |

**用途**：IC 判 SOP 01/02「没听清，请您再说一遍」（`asr_fail_streak` 累加）。

**流式与离线走同一套三档** —— 都用离线/流式各自的 `confidence.avg`，
用户不用记「流式看这个线、离线看那个线」。

### 完 `turn_complete_confidence` —— 本轮说完的把握

| 值 | 条件 |
|---|---|
| `HIGH` | **已拍板**：收到 `2pass-offline`，或 `turnsense` 判 `complete` |
| `MEDIUM` | 本段发生过 VAD 切分（知道有边界），但还没拍板 |
| `LOW` | 有转写、正在说，还没结束 |
| `NONE` | 无转写 |

**用途**：IC 判 `ANSWER`（`turn_complete == HIGH` 才回答）。

---

## 三、`transcript` —— 本轮说了什么

- 流式阶段：`2pass-online` 是**增量片段**（`"今天的"` / `"说出去看"`），
  **自己拼接**（服务端不发累积文本）
- 收到 `2pass-offline`：用最终结果**覆盖**
- 新一轮开始：清空

---

## 四、清空机制

| 字段 | 由什么驱动 |
|---|---|
| `说` `抢` | **段关闭即返回 `NONE`**（读时现算，不需要外部调用） |
| `完` `信` `text` | `expire_if_idle()` —— **必须由 `session.run_tick` 周期性调用**（50ms） |

⚠️ **`完`/`信`/`text` 为什么必须挂 tick**：清空逻辑若只放在「读快照」时，
而生产里唯一读快照的时机是**收到新 ASR 消息** —— 那么「说完一句就静音」的
场景下两者**永不相遇**，三个字段会**永远挂着**（实测：静音后 `完` 恒为 `HIGH`）。

这与 `_check_audio_watchdog` 是**同一类坑**：那个也不能挂在 `_periodic_diag`
（由 `on_audio` 调用，收不到音频时自己就不跑）。

### 保留时长

```
TURN_HOLD_MS = 1000   # 可用 ORCH_ASR_TURN_HOLD_MS 覆盖
```

从**段关闭**那一刻起算。

---

## 五、实测参考

两个服务端的字段差异（见 `asr/client.py` 的 `parse_confidence` /
`parse_online_confidence`）：

- `Fun-ASR-deploy`（Python）：online **和** offline 都发 `confidence` 对象
- `asr-2pass`（C++）：只有 offline 发该对象；online 走标量 `online_confidence`

**同一个 offline 帧里两个字段可以同时存在**，所以解析是「先标量、后对象」
的有序回退，不是二选一。

部分结果延迟实测 **950–1450ms**（见 `tests/README.md`）。
