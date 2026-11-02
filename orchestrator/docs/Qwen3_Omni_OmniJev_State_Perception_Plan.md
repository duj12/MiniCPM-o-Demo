# Qwen3-Omni 单模型 Omni-Jev：Social State 感知与输出方案

> 目标：一个多模态模型同时生成自然语言交互状态描述，并为 `Social_State_Schema_v2_1_Lite.md` 的每个枚举槽位选择当前取值；
> 自由文本槽位继续生成开放式描述。**字段、枚举与空值语义以该 Schema 为唯一权威，本文只讨论实现路径。**
> 当前硬件：2 × RTX 3090 24GB。
> 当前基线：用户已部署 Qwen3-Omni-30B-A3B。
> 日期：2026-11-02（同日晚修订：同步 Schema 2.1-lite，引入 E0/E1/E2 编码分级）。

---

## 1. 先固定 Omni-Jev 的定义

这里的 **Omni-Jev 不是 Agent 决策器**，不接收额外的 Q1～Q10 问题，也不输出“是否打断”“是否主动交互”“下一步做什么”等策略结论。

它只做一件事：**把原始音视频感知结果转成一份完整的 Social State。**

每轮输入：

- 当前原始视频帧/视频窗口；
- 当前原始音频窗口；
- 可选 ASR 文本及其时间范围（用于对齐，不作为 EMOTION 的视觉证据 —— 见 §2.3）。

⚠️ **上一轮 State Snapshot 不进入 prompt。** 状态延续由持续 KV 上下文 + State Engine 在模型外负责。
显式回灌上一轮状态会双重计入，且会诱发 stale carry —— 编排侧实测过：模型倾向于照抄自己前几轮的输出，
把"没变化"当成结论（06_1 的 `ATTENTION` 卡在"面向屏幕"就是这么来的）。

每轮输出（**形态由编码决定，见 Schema §4**）：

1. `state`：8 个顶层字段；
2. 每个可枚举子槽位的取值（E2 额外带 `confidence`，生成式阶段一律 `null`）；
3. 每个自由文本子槽位的自然语言内容；
4. `description`：可选，**只在 E2 / 离线评测里出现**。

**不输出**：字段级 `status`、`evidence`、`observed_at_ms` / 时间戳、每轮的 `schema_version`（E0 / E1），
以及 `questions` / `decisions` / `option_id` / Interaction Policy 的结论。后续 Agent 只消费 State，再自行执行策略。

## 2. 推荐架构：一个 Omni Backbone + 两类输出能力

**推荐长期结构：Qwen3-Omni Thinker 共享多模态 Backbone，保留语言模型输出能力，并增加一个轻量 State Head。**

```text
                   Audio + Video + ASR + Previous State
                                  │
                                  ▼
                       Qwen3-Omni Thinker
                shared multimodal encoder / Transformer
                                  │
                        shared hidden representations
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
                Language Head                State Head
                    │                           │
         自由文本 description              枚举槽位分类
         FEATURE / ENV / POSITION 文本       单选：softmax
         ACTION 事件说明                     多选：sigmoid
         State JSON 序列化                    每槽位置信度
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                 Unified Social State JSON Response
```

这是一个模型包、一套多模态 Backbone、一次音视频 Prefill。输出端分为语言生成分支和状态分类分支，不是两个模型串行调用。

### 2.1 Language Head 负责什么

- `description`（Schema 里是**单条短句**，可选；不要拆成 summary / visual / audio 三段）；
- FEATURE 衣着、配饰等长尾外观文本；
- ENV 环境要素描述；
- POSITION 与物体的相对位置描述；
- ACTION 事件的自然语言补充（`count` 在过渡期也由这里给）；
- E1 / E2 载荷的序列化。**E1 是固定的 `KEY: a|b|c` 文本，不需要模型输出 JSON 括号。**

### 2.2 State Head 负责什么

State Head 只负责封闭枚举及可分类槽位，不负责开放式文本。

| State 字段 | State Head 负责的槽位 | 仍由 Language Head 生成 |
|---|---|---|
| ATTENTION | `orientation`、`gaze` | — |
| ACTION | `events[].type`（多标签，sigmoid） | `count`（过渡期，见下）、事件自然语言补充 |
| EMOTION | `label`（融合）、`visual`（**只吃视觉 token**）、`conflict` | — |
| FEATURE | 暂不分类（长尾组合不值得多头） | `appearance`、`accessories` |
| BEHAVIOR | `posture`、`motion`、`activity` | `duration_s`（过渡期，见下） |
| RELATION | `interaction`、`target`；`person_count` 优先由检测器直接给 | — |
| ENV | `scene`、`lighting` | `objects` |
| POSITION | `region` | `relative_positions` |

> ⚠️ **过渡期例外**：`ACTION.events[].count` 与 `BEHAVIOR.duration_s` 在 State Engine 接入前**必须由模型生成**
> （SOP07_2 的摇头次数、SOP10_2 的远离秒数只有模型的观测窗口能给）。Schema §5.3 已把这个交接写清楚：
> 引擎上线后模型可省略，字段名和单位不变，下游无需区分来源。

每个单选槽位输出类别 logits，经 softmax 得到类别分布；每个多标签槽位输出各标签 logits，经 sigmoid 得到标签分数。置信度必须在独立验证集上校准。自由文本不伪造一个数值 confidence。

### 2.3 EMOTION 的视觉证据隔离：只管 `visual` 槽位

2.1-lite 把 EMOTION 定为**音视频融合**（表情 + 语调 + 说话内容语义），但保留了一个纯视觉旁证槽位 `visual`。
所以隔离要求的作用对象从"整个 EMOTION"缩小到**一个槽位**：

| 槽位 | 允许的证据 |
|---|---|
| `label` | 表情 + 语调 + 说话内容语义（融合） |
| `visual` | **只允许视觉 token / 视觉时间特征**，不得读取音频或 ASR 语义 |
| `conflict` | 由 `visual` 与语音侧极性比较得出 |

即使共用 Backbone，`visual` 的 slot-query 也应只 cross-attend 视觉 token。若当前 Qwen3-Omni 的内部融合结构
不便直接隔离，至少在训练与评测中做三件事：**音画冲突样本**、**视觉-only 对照**、**输入模态屏蔽测试**
（同一段视频、替换音频内容后 `visual` 必须保持不变）。

`visual` 的隔离质量直接决定 `conflict` 的可信度 —— 语音一旦污染 `visual`，冲突检出的上界就没了。
另注意：`label` 融合用到的"说话内容语义"来自模型**直接听到的音频**，不依赖 ASR 转写，因此不增加链路等待。

### 2.4 ACTION / BEHAVIOR 的时间语义

ACTION 是短时事件，BEHAVIOR 是持续状态。对短动作计数、开始/结束和持续时间，不能仅靠模型对单帧的语言描述。

推荐：State Head 使用视频时间序列特征；检测/跟踪与 State Engine 负责事件去重、跨窗口合并、duration 和 age 计算。
模型给事件类型即可，派生时间由 State Engine 算。

**过渡期例外（现在就要做）**：State Engine 尚未接入，所以 `ACTION.events[].count` 与 `BEHAVIOR.duration_s`
必须由模型生成（Schema §5.3 已固定字段名与单位）。引擎上线后模型可省略这两个槽位，下游无需改。

**一条必须写进 prompt 的约束**：1~2 秒的短暂动作，即使解码时已经结束，只要落在观测窗口内就要报告。
描述触发点在断句或心跳之后，等生成时动作往往已经结束 —— 不写这一句，SOP07 全漏。

---

## 3. 置信度应该怎么产生

> 命名说明：本章的 **L0 / L1 / L2 是"置信度来源"的等级**；Schema §4 的 **E0 / E1 / E2 是"载荷编码"的等级**。
> 两者相互独立 —— 例如 Phase 1 是「E1 编码 + 无置信度（L-）」。

不要让模型在 Prompt 中随手写 `confidence: 0.92`。那只是模型自评文本，不代表校准概率。

### L0：Prompt 自报置信度（不推荐）

可以用于最早的 JSON 格式 Demo，但不能用作线上阈值、策略门控或模型横向比较。

### L1：从现有 LM Head 读取枚举候选 logits（过渡方案）

做法：为每个枚举槽位准备固定候选标签，让 Thinker 在固定结构位置输出枚举 token / 短标签代码；在该位置读取候选 token logits，只在该槽位允许的类别集合内归一化。

注意：

- 短代码必须在 tokenizer 中验证为单 token；否则要正确计算候选标签的条件序列概率，不能简单平均 logprob。
- 多标签槽位要按每个标签分别建二分类概率，不能用单选 softmax。
- 需要确认当前 vLLM-Omni Streaming Video WebSocket 路径能否暴露这些位置的 logits/logprobs；当前公开 `response.text.delta` 主要是文本增量，不能假设它已经提供 State logits。
- 即使拿到 logits，仍然需要按字段做概率校准。

L1 不增加独立分类模型，但需要在当前模型输出/服务层增加候选打分读取逻辑。适合快速验证“现有 Thinker 能否直接承担枚举分类”。

### L2：共享 Backbone 的专用 State Head（推荐目标）

在 Qwen3-Omni Thinker 的 hidden states 上增加小型 slot-query / classification heads：

1. 为每个枚举槽位学习一个 slot embedding / query；
2. query 从相关模态和时间窗口的 hidden states 中读取证据；
3. 单选槽位输出 `K` 维 logits；
4. 多标签槽位输出 `K` 个 sigmoid logits；
5. 经过温度缩放或其他校准后，输出类别和 confidence。

优点：

- 分类不需要自回归生成一串枚举文本；
- 多个 State 槽位可在同一批 hidden states 上并行计算；
- confidence 来自明确的类别分布，而不是模型自述；
- Language Head 仍然保留自由文本描述能力。

代价：需要修改模型实现、训练代码和 vLLM-Omni 的 Thinker stage 输出/后处理。它不是简单加一段 Prompt 就会自动出现的能力。

---

## 4. 推荐实施顺序

### Phase 0：固定契约

先以 `Social_State_Schema_v2_1_Lite.md` 为唯一权威，删除原先“10 个问题 / decisions / option_id”的接口设计。
每轮只输入媒体 + ASR；**不输入 previous_state**（见 §1）。每轮输出 8 字段 state，形态由编码决定。

关键是接受 Schema 的**定义与编码分离**（Schema §4）：现在用 E1，将来切 E2，**字段与枚举不动**。

### Phase 1：当前 Qwen3-Omni 生成式基线

使用现有 2 × RTX 3090 服务，不替换模型、不先改模型结构：

- 原始音频 / 视频输入；
- 固定 system prompt（字段与枚举白名单放在这里，每轮注入带不动类别 —— 编排侧实测过）；
- 单次生成 8 字段状态，编码用 **E1 扁平槽位文本**（见 Schema §4.2）。

这一阶段**不输出** `confidence` / `description` / `schema_version` / `evidence` / 时间戳。
重点验证：枚举合法率、字段完整率、每轮输出 token 数与端到端延迟，以及逐类 SOP 回放。
预算参考：**每轮 ≤150 token 输出、P95 ≤1.2s**（按编排侧 ~150 字 / ~1s 的生成式实测基线；
JSON + 逐槽位 confidence 会把输出量推到约两倍）。

### Phase 2：从 LM logits 获取枚举分数

如果 vLLM-Omni 当前推理路径能够在固定 State 槽位暴露候选 logits/logprobs，就先做 L1。用现有 Thinker 对每个枚举槽位的合法类别打分，生成类别与初始置信度；对自由文本继续正常生成。

这一阶段的目的是快速验证：在不增加新模型参数的前提下，现有 Qwen3-Omni 是否已经具备足够好的枚举区分能力。

### Phase 3：多任务 SFT + State Head

如果 L1 的类别区分度、延迟或校准效果不够，再加专用 State Head。训练样本格式：

`(audio, video, asr_turn) -> (state_labels, free_text_slots)`

建议损失：

`L = λ_text * L_text + λ_single * L_single_enum + λ_multi * L_multi_label + λ_event * L_event + λ_temporal * L_temporal`

其中：

- `L_text`：description 和自由文本字段的 next-token loss；
- `L_single_enum`：单选枚举槽位交叉熵；
- `L_multi_label`：多选槽位 BCE；
- `L_event`：ACTION 事件分类/检测损失；
- `L_temporal`：事件边界或行为阶段损失，若数据标注支持再加入。

可以从冻结大部分 Backbone + LoRA + 训练轻量 State Head 开始。先在足够显存的训练环境做完整试验；2 × RTX 3090 适合继续做现有推理验证，但不能预设一定能够稳定完成 30B-A3B 多模态 LoRA 全流程。

---

## 5. 流式 Prefill 与当前 vLLM-Omni 链路

继续复用已经部署的 Qwen3-Omni 和现有双卡推理服务。vLLM-Omni 当前公开的 Streaming Video Input API 支持持续推送 `video.frame` 和可选 `audio.chunk`，再由 `video.query` 触发文本输出；`response.text.delta` 可流式返回生成文本。stage-0 Thinker 开启 `enable_prefix_caching` 后，视频帧可进行增量 Prefill。

关键边界：

- 当前增量 Prefill 主要复用视频视觉前缀；查询时追加音频/文本后缀，不等于音频 chunk 已在同一路径完全增量 Prefill。
- 该接口的文本输出路径适合你的描述和 State JSON；不需要启用 Talker / Code2Wav。
- 现有 WebSocket API 返回文本增量，并不会自动输出自定义 State Head 的 logits。Phase 3 需要把 State Head 挂到 Thinker stage，在共享 Backbone 的同一次推理中计算 State 分类结果，并扩展 serving response / output adapter。
- State Head 需要与 Language Head 共享相同的媒体窗口、帧采样和时间戳；不能因异步输出分支看到不同证据而产生冲突。

建议在线时序：

```text
持续阶段：
  Camera -> 去重/采样 -> video.frame -> Thinker 视觉增量 Prefill
  Microphone -> audio.chunk buffer
  ASR -> partial/final

触发点（三类，编排侧已实现，不能只留 ASR endpoint）：
  A. 会话起点（IC 判 GREET，或 IC 直判 ANSWER 时首个带文本的 ASR） -> full（8 字段带细节）
  B. VAD 断句点                                                   -> turn
  C. 静默 10s 心跳                                                -> turn
     ⚠️ 用户不说话时 A/B 都不会发生；没有 C 就永远看不到"画面变了但没人说话"

每次触发：
  0. 帧新鲜度闸：距最近一次**真正送出**的视频帧 > 5s ⇒ 不触发，保留上一份快照
     （没有新画面时模型不会说"看不到"，而是按字段格式编造 —— SOP 尾巴上复现过两次）
  1. 固定本轮媒体窗口及 PTS
  2. 附加 ASR final（**不回灌 previous_state**，见 §1）
  3. 触发一次 video.query
  4. Language Head 生成自由文本槽位（description 仅 E2）
  5. State Head（Phase 3）在同一轮共享 hidden states 上输出枚举槽位与 confidence
  6. 输出适配层按 E1 / E2 编码合成完整 state
  7. Schema validator + State Engine 合并 / 计算 TTL
```

实现提醒：Phase 1 可直接消费 `response.text.delta`；Phase 3 需要自定义 stage / runner hook。不要假设只改 Prompt 就能让现有 WebSocket 自动支持并行 State Head。

---

## 6. 输出示例：E1（当前生产）与 E2（目标）

**E1 —— 扁平槽位文本**（当前编排侧采用，零改动接入）：

```text
ATTENTION: facing_camera|at_camera
ACTION: shake_head*3
EMOTION: dissatisfied|visual=dissatisfied|conflict
FEATURE: 青年女性，灰色毛衣，长裤|黑框眼镜
BEHAVIOR: standing|stationary||14s
RELATION: 2|talking|another_person
ENV: exhibition|normal|玻璃展柜,展示海报
POSITION: center|位于展台左侧,旁边有一人
```

**E2 —— JSON**（State Head 目标形态）：

```json
{
  "schema_version": "2.1",
  "state": {
    "ATTENTION": {
      "orientation": {"label": "facing_camera", "confidence": null},
      "gaze": {"label": "at_camera", "confidence": null}
    },
    "ACTION": {
      "events": [{"type": "shake_head", "count": 3, "confidence": null}]
    },
    "EMOTION": {
      "label": {"label": "dissatisfied", "confidence": null},
      "visual": {"label": "dissatisfied", "confidence": null},
      "conflict": true
    },
    "FEATURE": {"appearance": "青年女性，灰色毛衣", "accessories": ["黑框眼镜"]},
    "BEHAVIOR": {
      "posture": {"label": "standing", "confidence": null},
      "motion": {"label": "stationary", "confidence": null},
      "activity": {"label": "observing_object", "confidence": null},
      "duration_s": 14
    },
    "RELATION": {
      "person_count": 2,
      "interaction": {"label": "talking", "confidence": null},
      "target": {"label": "another_person", "confidence": null}
    },
    "ENV": {
      "scene": {"label": "exhibition", "confidence": null},
      "lighting": {"label": "normal", "confidence": null},
      "objects": ["玻璃展柜", "展示海报"]
    },
    "POSITION": {
      "region": {"label": "center", "confidence": null},
      "relative_positions": ["位于展台左侧"]
    }
  }
}
```

> 两例表达**同一份状态**：E2 → E1 是无损降级（丢 `confidence` / `description` / `schema_version`），
> E1 → E2 是补 `null` 后升格。枚举白名单、空值语义（`unknown` / `null` / `[]`）一律以
> `Social_State_Schema_v2_1_Lite.md` 为准；本文不重复定义。
> `confidence` 全为 `null` 是因为生成式阶段不许自报数字（§3），State Head 上线后换成实数，结构不变。

---

## 7. 训练数据与评测

### 7.1 标注数据

每个训练样本至少包含：

- 原始视频窗口、音频窗口和统一媒体时间戳；
- ASR 文本（可选，但与音频分开保存）；
- 8 字段的人工核验标签；
- 每个枚举子槽位的类别标签；
- 多标签槽位的正负标签；
- 自由文本描述；
- 对不可见、不确定、未检测到、无、不适用、过期的清晰区分；
- ACTION 事件时间范围与 BEHAVIOR 连续状态片段（若要求时序监督）。

不要把模型生成的 confidence 当作训练真值。训练真值是类别标签；置信度由 logits 和验证集校准产生。

### 7.2 数据切分与困难样本

- 以完整 session / 人员 / 视频为单位切分 train/val/test，避免邻近帧泄漏；
- 保留 SOP 06～11 原有用例；
- 补充遮挡、背对、视线游移、多人切换、短时动作、画面/语音情绪冲突、音画不同步、帧丢失等样本；
- EMOTION 增加同视频、相同脸部、不同语音内容的对照：**`visual` 槽位必须不变**，`label` 允许随之变化，
  且语音与表情极性相反时 `conflict` 应为 `true`（这是 §2.3 隔离设计的直接验收）。

### 7.3 核心指标

| 指标 | 用途 |
|---|---|
| 每枚举槽位 Accuracy / Macro-F1 | 分类能力 |
| 每枚举槽位 ECE / Brier Score | 置信度校准（仅 Phase 2 之后有意义） |
| 多标签 mAP / F1 | `ACTION.events`、`accessories` 等数组槽 |
| Description Faithfulness | 自由文本是否忠于可观察证据 |
| State 完整率 / **E1 解析合法率** | 协议遵循（E1 用 `parse_state` 正则直接验） |
| ACTION event recall / precision（含 `count` 误差） | 瞬时事件检出 |
| BEHAVIOR motion / activity F1，`duration_s` 误差 | 持续状态与时间语义 |
| **音画冲突检出率**（`conflict` 的 precision / recall） | 唯一无法由 `label` / `visual` 推导的信号，必须单独验 |
| **无新帧时的编造率** | 帧新鲜度闸的兜底：停止送帧后仍输出具体内容即为失败 |
| P50 / P95、TTFT、TPOT、每轮输出 token 数 | 实时性能 |

---

## 8. 双 RTX 3090 部署策略

1. 保留当前已运行的 Qwen3-Omni checkpoint、vLLM-Omni 版本、量化方式和 TP 配置，避免一次性更换底座与 Serving。
2. 输出仅需文本，因此优先 Thinker-only 路径，不加载 Talker / Code2Wav（以你当前实际部署验证为准）。
3. 初期并发保持 1，先记录从 ASR endpoint 到首个有效 State 字段、到完整 JSON 的 P50/P95。
4. 流式视频帧按相似度/事件驱动采样，避免无差别把所有摄像头帧送进 Prefill。
5. 固定 system prompt、Schema 和字段顺序，尽可能复用相同视觉前缀与历史上下文；每轮历史快照只保留必要 State，不无限堆积对话。
6. Phase 3 的自定义 State Head 与 Thinker 放在同一模型服务中，复用媒体表征和设备，不启动第二个模型进程。
7. 若双卡显存和调度限制导致 head 输出或隐藏状态缓存占用明显增加，优先保存 slot-query 所需的 pooled features，不要长期保留所有高分辨率 token hidden states。

---

## 9. 实施优先级与验收

| 阶段 | 工作 | 验收标准 |
|---|---|---|
| P0 | 接口改为「8 字段 State」，编码用 **E1** | 每轮 ≤150 token；`parse_state` 解析合法率 100%；无 decisions |
| P0 | 固化枚举与空值语义（Schema 为唯一权威） | 枚举值全部合法；`unknown` / `null` / `[]` 不混用 |
| P0 | 接上三类触发点 + 帧新鲜度闸 | 静默用户也能被描述；**停帧 5s 后不再产出新描述** |
| P0 | 当前 Qwen3-Omni Prompt baseline | 字段完整性、延迟、逐类 SOP 回放基线数据 |
| P1 | 验证 Thinker LM logits / logprobs 提取 | 获得候选类别原始分数并测其区分度 |
| P1 | SOP 回放与概率校准 | 每个枚举槽位有 Macro-F1、ECE/Brier 报告 |
| P2 | 多任务 SFT + State Head | 共享 Backbone 输出自由文本 + State 分类；**编码切到 E2** |
| P2 | vLLM-Omni Thinker stage 接入 State Head | 一次媒体 Prefill 返回文本和枚举 logits；`confidence` 从 `null` 变为实数 |

**最终推荐：先不更换 Qwen3-Omni。把 Omni-Jev 定义为“多模态描述生成 + Schema State 分类”的单模型多任务感知器。
先跑生成式基线（E1 编码），再从 LM logits 验证可行性，最终在同一 Thinker Backbone 上增加专用 State Head，
以获得可靠的枚举类别和经过校准的置信度（E2 编码）。**

**全程共用一份 Schema（`Social_State_Schema_v2_1_Lite.md`）：字段名、枚举取值、空值语义在三个阶段完全一致，
升级只切换编码与值的来源。** 任何需要改 Schema 本身的改动都属于协议变更，必须单独评审。

---

## 10. 参考资料

- Qwen3-Omni 官方仓库：https://github.com/QwenLM/Qwen3-Omni
- vLLM-Omni Qwen3-Omni 在线服务：https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/qwen3_omni/
- vLLM-Omni Streaming Video Input API：https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/video_stream_api/
- vLLM-Omni Prefix Caching：https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/
