# ASR 五个状态量 —— 编排层的消费约定

**2026-10 起，这五个量全部由 ASR 服务端计算**，放在每条消息的 `state` 里
下发（取值规则的权威定义见 `asr-2pass/docs/websocket-api.md` §4 与
`Fun-ASR-deploy/docs/websocket-api.md` §4）。编排层**不再自己归纳** ——
`orchestrator/asr/client.py` 的 `AsrStateTracker` 现在只是：

1. 把服务端的 `state` 原样缓存成 `AsrState`（档位归一成大写四值）；
2. 一轮结束后分两拍把快照清零（见第三节）。

下发：`AsrDisplay.state`（UI）/ `apply_vad` `apply_asr`（InteractionCore）。

---

## 一、字段（服务端语义摘要）

| 字段 | 取值 | 含义 |
|---|---|---|
| `user_speaking_confidence`（说） | `HIGH` / `MEDIUM` / `NONE` | 此刻是否在出声 |
| `barge_in_confidence`（抢） | `HIGH` / `MEDIUM` / `LOW` / `NONE` | 此刻是否在抢话 |
| `transcript` | string | 本轮累计转写（服务端维护：流式拼接、离线覆盖、新轮清空） |
| `asr_confidence`（信） | `HIGH` / `MEDIUM` / `LOW` / `NONE` | 本轮转写置信度 |
| `turn_complete_confidence`（完） | `HIGH` / `MEDIUM` / `LOW` / `NONE` | 本轮是否说完 |

服务端的具体档位规则（阈值、拍板条件、抢话档位）见上面两份
`websocket-api.md` §4 —— **以那里为准**，本文不复述，避免两处漂移。

`state` 里还会有 9 个感知字段（`language` / `emotion` / `snr_db` /
`noise` / `overlap` …）。编排层**默认打开**（`AsrConfig.enable_perception =
True`，首帧连同 6 个逐项开关一起下发），所以线上 `state` 是 14 个 key。

- 部署侧想关：`ORCH_ASR_PERCEPTION=0`（`orchestrator/config.py` 的
  `Settings.asr_perception`）→ 首帧 `enable_perception=false`，`state` 回到
  5 个 key，服务端也不再跑感知推理。
- **不让前端单会话改**：服务端模型是全局加载的，逐会话开关只会让排查时
  说不清"到底开没开"。
- 编排层目前对这 9 个字段**只透传、不消费**：`AsrState` 只取那 5 个基础
  字段 + `transcript`，其余 key 原样丢弃（不会报错，也不进 IC/UI）。

**档位归一**：服务端下发什么就是什么，只有大小写与未知值做归一
（`high` → `HIGH`，缺失/`null`/未知 → `NONE`）。客户端不推导档位。

---

## 二、为什么还要在客户端清一次

服务端**只在"有消息"时带 `state`**。一句话说完（`2pass-offline`）之后
它不会再主动下发任何消息，而 IC 需要知道"状态已经清了" —— 不清的话
它会永远停在最后一条快照上（实测：一轮结束后 `说`/`抢` 一直 `HIGH`）。

所以 `session.run_tick` 每 50ms 调一次 `AsrClient.expire_if_idle()`，
由它把**已结束的那一轮**推过清零的两个拍子。

---

## 三、清零：两拍，没有时间窗口

原来的 `TURN_HOLD_MS`（默认 1s，可用 `ORCH_ASR_TURN_HOLD_MS` 覆盖）
**已删除**。现在的时序完全由 tick 驱动：

| 拍 | 触发 | 快照 |
|---|---|---|
| 0 | 收到 `2pass-offline` | 服务端原值：`说`/`抢`=HIGH，`完`=HIGH，带本轮转写 |
| 1 | 下一次 tick（~50ms） | `说`/`抢` → `NONE`；`完`/`信`/`transcript` **保留** |
| 2 | 再下一次 tick（~50ms） | 全部清空 |

**拍 1 不能省**：IC 的 `fresh_turn → ANSWER` 依赖"已经安静下来、但本轮
已拍板"这份快照；只有 `offline` 那一拍的话，`完=HIGH` 会被 IC 同拍的
`advance_*` 吃掉（实测 `turn=NONE`、只回 HOLD）。
见 `tests/test_omni_describe_stage.py` 的 `_ASR_SETTLED` 说明。

**例外**：

- `turnsense=incomplete` → 段关了但话没说完，两拍都**不发生**，
  等后半句（否则 IC 会以为一轮结束，提前回答）。
- 收尾帧（`is_final=true` 且空文本）→ **立即**全部清空。
  服务端那一拍里的值（可能 `说=MEDIUM`）只是兼容残留，不透传。
- 新的一段开始（收到 `2pass-online`）→ 取消待清零，快照换成新消息的 `state`。

---

## 四、两个解析细节（仍与编排层有关）

**置信度字段形态不同**（`asr/client.py` 的 `parse_confidence` /
`parse_online_confidence`）：

- `Fun-ASR-deploy`（Python）：online **和** offline 都发 `confidence` 对象
- `asr-2pass`（C++）：只有 offline 发该对象；online 走标量 `online_confidence`

**同一个 offline 帧里两个字段可以同时存在**，所以解析是「先标量、后对象」
的有序回退，不是二选一。这只影响日志/展示，`信` 档位本身由服务端算好。

`turnsense` 消息的 `probabilities` 也有两种形态（C++ 是数组 + `prediction_id`，
Python 是 dict），由 `extract_turnsense()` 统一。

部分结果延迟实测 **950–1450ms**（见 `tests/README.md`）。
