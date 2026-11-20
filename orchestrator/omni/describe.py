"""两阶段 OmniLLM 描述 —— **提示词与判据集中放这里**。

## 两阶段是什么

| 阶段 | 何时 | 指令 | 期望输出 |
|---|---|---|---|
| `full`  | 会话起点（IC 判 **GREET**，或 IC 直判 ANSWER 时**首个带文本的 ASR 结果**） | `FULL_INSTRUCTION` | 8 个字段**带细节**（见 `STATE_KEYS`）。只发这一次，进 Agent 的「会话起点视觉背景」 |
| `turn`  | VAD 断句点 / 10s 心跳 | `TURN_INSTRUCTION` | **同样是 8 个字段全部输出**，但每个值压到最短 |

两阶段**共用一条连接、一个 system prompt**，靠 ``trigger_reply(text=…)``
注入不同指令来区分 —— 不中途 re-init（那会把已累积的视听 KV 上下文丢掉，
首字要重新热起来），也不开第二条连接（106:8006 上白白多占一路）。

## 提示词怎么分

* `STATE_SYSTEM_PROMPT` / `DESCRIBE_SYSTEM_PROMPT` —— **字段契约 + 口吻**
  （8 个键、每行 `KEY: 内容`）。system prompt 每轮都作为前缀重发，
  所以"描述哪些字段"必须在这里；
* `FULL_INSTRUCTION` —— 会话起点：8 个字段，**允许展开细节**（这份要当背景）；
* `TURN_INSTRUCTION` —— 每轮：8 个字段，**每个值压到最短**（成本在这里）。
* `SHORT_FULL_INSTRUCTION` / `SHORT_TURN_INSTRUCTION` —— **接了 state head 之后**
  （`ORCH_STATE_HEAD=1` 且 `ORCH_TEXT_SHORT=1`，且 `ORCH_STATE_SLOTS` 为空
  = E2 出全部枚举槽位）改用的短版：只输出 `FEATURE` / `ENV` / `POSITION` /
  `DESCRIPTION` 4 行。没开 state head 就**不要**用它 —— 8 个键里砍掉 5 个
  而没人补，等于把状态丢了。
* `partial_instruction(stage, slots)` —— **槽位白名单**模式
  （`ORCH_STATE_SLOTS` 非空，默认 `attn.gaze,beh.activity,rel.person_count,
  rel.interaction`）：保留 8 行形状，只把 E2 已经答了的项交代成"别写"。
  为什么不直接砍成 4 行：见 `partial_instruction` 的 docstring。

关键约束：`system_prompt` 只在 `session.init` 时定一次（`input.append` 没有
逐轮 prompt 字段）。所以"这一轮要什么"只能靠注入指令区分 —— 但也**只能
到这一步**：注入指令带不动"描述哪些类别"，所以类别得留在 system prompt 里。

## 为什么每轮也输出全量（2026-10-28 改）

更早的做法是「全量 + 增量（只报变化的字段）、编排侧合并成快照」。105 影子
8101 上 SOP 06–11 十个片段、138 次描述的实测证明它有**两个结构性毛病**，
都不是调 prompt 能修的：

1. **stale carry**：增量不报的字段会**一直**沿用旧值。06_1 里人物已经背对
   镜头，`ATTENTION` 却一路是「面向屏幕」—— 模型每轮都判断"注意力没变"，
   于是这个字段根本不输出，编排侧就无从更新。
2. **抽象元指令跟不动**：138 次里有 4 次把「无变化」当成**字段值**写出来
   （`ATTENTION: 无变化` / `BEHAVIOR: 无变化`），且**全部发生在增量轮、
   全量轮一次都没有**。

同一批数据里，**格式跟随是好的**：32 次全量轮 **32/32 都输出了完整 8 键**。
⇒ 让模型每轮照抄「输出这 8 行」这个**具体**指令，比让它判断「什么算变化」
这个**抽象**指令可靠得多。代价是每轮多生成一些字（实测全量中位
1067ms / 149 字 vs 增量 277ms / 17 字）。

### ⚠️ 「每轮把值压短」试过两版，**都没效果**（2026-10-28，105 影子 8101）

| 版本 | turn 平均字数 | turn 平均耗时 |
|---|---|---|
| ① 抽象要求「一般不超过 12 个字」 | 147 字（n=47） | 1003ms |
| ② 逐字段预算 + 一份可照抄长度的示例块 | 151 字（n=18） | 979ms |
| 同会话的 full（带细节） | 146~147 字 | ~1030ms |

**两版都与 full 一模一样** —— 逐轮注入压不动详细程度。最可能的原因是
**上下文自模仿**：每轮的会话上下文里堆着自己前几轮那份详细输出，模型照抄
自己的风格；注入指令只是其中一句。这与下面「system 带得动、注入带不动」
是同一条规律的另一种表现。

⇒ 想让每轮真的变短，只能换 **system prompt**（那就得关掉重连一次，即
早先 `_switch_delta_persona` 那套机制）；`TURN_INSTRUCTION` 留着是为了
**要求全量 8 键 + 禁止「无变化」**（这两条它做得到，见上面的实测）。
成本按 **~150 字 / ~1s 每轮**估，非阻塞。

## 顺带保留的一条实测约束

「必须 system 才可以吗，不能是每一轮的 prompt 限定输出吗？」—— 2026-10-08
在 105 上试过三版：system 摘掉类别清单、只放在每轮注入里 ⇒ **全量也塌了**
（151 字普通段落，注入的清单根本没被采纳）。
**每轮注入带不动"输出哪些类别"，system prompt 带得动** —— 这条约束现在还
适用；而且 2026-10-28 又测出**详细程度也一样带不动**（见上一节两张表），
所以 `TURN_INSTRUCTION` 实际只负责两件它做得到的事：**要求全量 8 键**、
**禁止把「无变化」当字段值**。

## 键值契约带来的变化（相对自由文本）

1. **每轮与起点的差别只在"详细程度"**，不在格式 —— 两阶段都是同一套
   `KEY: 值` 行格式，Agent 侧看到的东西形状恒定。
2. **`max_new_tokens` 从"不能用"变成"可以用"**：老版本担心截在半句、
   而描述是**原样**发给 Agent 的；现在 `parse_state` 按行解析，截断最多
   影响**最后一行**，不会污染整份快照。
   ⚠️ 代码的真实行为是「末行即使只有半句，也会**带着半截值**被合并，盖掉
   那**一个**字段」，不是"整行丢掉"。因为每轮都是全量、下一轮必然重写全部
   字段，这个误差只存活一轮。

## system prompt 与指令的分工

* system prompt 决定**口吻**（描述而非对话）与**描述哪些字段**；
* 注入指令决定**这一轮的详细程度**（起点带细节 / 每轮压到最短）。

## 与 Agent 载荷的关系

描述文本经 `OmniDescription` 事件落进
`orchestrator/interaction/downstream.py`：先 `parse_state()` 解析，
按键**合并**进缓存快照，再用 `render_state()` 渲染成完整的 8 行文本，
然后由 `agent_sink.py` **立刻**以「视觉轮」（ASR 空的 `on_answer`）推给 Agent
—— Agent 只存不答，覆盖写 `scene.visual_background`，随 system prompt 的
「会话状态」段进入每一轮 LLM 调用。
**VLM 与 ASR 不同步**（2026-10-28）：正常话轮只带 ASR、``content.VLM`` 留空，
两边各自就绪各自发。**只在 `response.done`（生成完的整段）取值**，绝不用
`response.output.delta` 的 token 流 —— 那是半句话（见 `omni/client.py:_emit`）。
"""
from __future__ import annotations

import re

#: 字段名（有序）—— 渲染顺序、解析白名单都以此为准。
#:
#: 对应 `docs/交互策略SOP视频测试用例.md` 的 06–11：
#: ATTENTION=06 / ACTION=07 / EMOTION=08 / FEATURE=09 / BEHAVIOR=10 / RELATION=11，
#: 外加 ENV、POSITION 两个原有类别（保留它们，否则环境与相对位置会丢）。
STATE_KEYS: tuple[str, ...] = (
    "ATTENTION", "ACTION", "EMOTION", "FEATURE",
    "BEHAVIOR", "RELATION", "ENV", "POSITION",
)

_KEY_SET = frozenset(STATE_KEYS)

#: 会话初始化时的 system prompt —— **字段契约 + 口吻**。
#:
#: 三段实测（2026-10-08，105）换来的口吻约束，别删：
#:
#: 1. system prompt 写成一句抽象的"你是描述助手" ⇒ 会话刚起 2.5s、
#:    视频帧才 ~4 帧时点火全量描述，模型回了 `'好的，现在开始录音。'`。
#:    **抽象人设定不住口吻** —— 得有"视频监控/行为分析助手"这个身份。
#: 2. 字段清单**必须在 system prompt 里**：摘掉它、只放在每轮注入里，
#:    全量轮也不分条了（151 字普通段落）。
#: 3. "不要与用户对话"这条护栏是**口吻锚**，删掉模型会开始聊天。
STATE_SYSTEM_PROMPT = """你是一个视频监控/行为分析助手。请综合画面与语音，输出视频的结构化状态。
要求：只描述画面中实际可见、语音中实际可闻的信息；不确定的写"不可见/不确定"，绝不编造。
输出格式：每行一个字段，写成「字段名: 内容」。字段名只能用下面这 8 个英文大写词，顺序也按下表。

ATTENTION —— 交互注意力：只能取「面向屏幕(注视镜头) / 侧身或背对 / 注意力在别处 / 正在打电话 / 不可见」之一。一边看屏幕一边打电话时，写「正在打电话，时而看向屏幕」
ACTION —— 动作表达：只能取「点头 / 摇头(注明次数) / 竖大拇指 / 挥手 / 鼓掌 / 指向某物 / 无」之一；"说话"不是动作。1~2 秒的短暂动作即使已经结束也要报告，只有确实没看到任何动作才写「无」
EMOTION —— 视觉情绪：只能取「满意 / 高兴 / 愤怒 / 不耐烦 / 平静 / 不可见」之一。只看表情、神态与姿态，绝不参考人物说了什么、语气如何。判据是眉毛与嘴角：皱眉、眉头紧锁、嘴角下压、目光锐利 ⇒ 不耐烦或愤怒；微笑、嘴角上扬 ⇒ 满意或高兴。脸部清晰可见时不要写「不可见」
FEATURE —— 自身特征：性别、年龄段、穿着（上衣/下装颜色款式、配饰）
BEHAVIOR —— 连续行为：只能取「站立 / 坐 / 行走 / 转身 / 靠近 / 远离 / 保持 / 离开画面」之一，并写明持续了多久（如「持续远离 6 秒」）
RELATION —— 用户关系：单人；多人时写人数、同伴关系（同行/陪同/主客/路人）、社交距离、谁是主要交互对象
ENV —— 环境：场景类型、其他人、环境要素（物品、光线、背景特征）
POSITION —— 位置：人物与画面中主要物体的相对位置；被指向或拿起的物体

示例（只示意格式，内容以实际画面为准）：
ATTENTION: 面向屏幕，注视镜头
ACTION: 无
EMOTION: 平静
FEATURE: 青年女性，灰色短袖上衣
BEHAVIOR: 站立，保持不动
RELATION: 单人
ENV: 室内展厅，光线明亮
POSITION: 站在展台右侧

不要与用户对话，不要寒暄、提问、回应或反问，只输出上述字段。"""

#: `main.py` 装配 omni 时读的常量名 —— 保留它是为了不动调用点。
DESCRIBE_SYSTEM_PROMPT = STATE_SYSTEM_PROMPT

#: 全量描述指令（以 ``trigger_reply(text=…)`` 注入）。
#:
#: 用户定调：「全量的是这个（demo 的 prompt）」—— 契约内容已经在 system
#: prompt 里，这里只交代"这一轮输出全部字段"。前面那半句是针对
#: 2026-10-08 走样（模型回 `'好的，现在开始录音。'`）的护栏。
FULL_INSTRUCTION = (
    "直接输出对当前画面与语音的结构化状态。"
    "不要确认、不要回应、不要以“好的/收到/现在开始”之类的话开头，从第一个字段名直接开始。\n"
    "8 个字段全部输出，每个字段一行，格式「字段名: 内容」。"
)

#: **每轮**描述指令（VAD 断句点 / 10s 心跳注入）。
#:
#: 与 `FULL_INSTRUCTION` 的差别**只在详细程度**：同样输出 8 个字段，但每个
#: 值压到最短。原因是成本 —— 生成耗时与输出长度几乎线性，而这条路径每轮
#: 都会走一次（全量中位 1067ms/149 字 vs 精简后目标 ~80 字）。
#:
#: 三条要求不能省，都是实测换来的：
#:
#: * **"8 个字段全部输出"** —— 这是全量方案的全部价值。此前"只报变化"的
#:   增量指令会让模型跳过它认为没变的字段，编排侧就一直沿用旧值（06_1 的
#:   `ATTENTION` 卡在「面向屏幕」就是这么来的）。
#: * **"绝不能写「无变化」"** —— 增量时代实测 4/138 次把「无变化」当成
#:   **字段值**写出来，污染快照且会一直留在载荷里。
#: * **短暂动作"即使已经结束也要报"** —— `ACTION` 的点头/摇头/竖大拇指是
#:   1~2s 的瞬时动作，而触发点在断句或心跳之后，等生成时往往已经结束。
#:   不写这一句，07 的用例就全漏了。
TURN_INSTRUCTION = (
    "本轮输出**当前完整状态**：8 个字段全部输出，每个字段一行「字段名: 内容」，顺序同上。\n"
    "**每个值只写判定结果，越短越好**（一般 2~10 个字），不解释、不展开：\n"
    "- ATTENTION：只写镜头朝向/注意力去向（如「面向屏幕」「侧身或背对」「正在打电话」）；\n"
    "- ACTION：只写动作名与次数（如「点头 2 次」「竖大拇指」「无」）；\n"
    "- EMOTION：只写一个词；\n"
    "- FEATURE：只写性别、年龄段、上衣（如「青年女性，深色长袖」），不写发型、配饰、下装；\n"
    "- BEHAVIOR：姿态 + 持续秒数（如「站立 6 秒」「持续远离 8 秒」）；\n"
    "- RELATION：人数 + 关系 + 主要交互对象（如「两人同伴，左侧为主要对象」）；\n"
    "- ENV：只写场景类型（如「室内走廊，有海报」），不列光线与物品清单；\n"
    "- POSITION：只写位置（如「画面中央」「玻璃门左侧」）。\n"
    "另外两条硬要求：\n"
    "- 字段值必须是**状态描述**，绝不能写“无变化”这种话；\n"
    "- 短暂动作（点头/摇头/竖大拇指/挥手/鼓掌/指向）即使已经结束也要报出，并注明次数。\n"
    "长度照下面这个示例写（内容以实际画面为准，不照抄文字）：\n"
    "ATTENTION: 面向屏幕\n"
    "ACTION: 无\n"
    "EMOTION: 平静\n"
    "FEATURE: 青年女性，深色长袖\n"
    "BEHAVIOR: 站立 6 秒\n"
    "RELATION: 单人\n"
    "ENV: 室内走廊\n"
    "POSITION: 画面中央\n"
    "不要与用户对话，不要确认，不要以“好的/收到”开头，从第一个字段名直接开始。"
)


#: **短文本**版起点指令 —— 仅在 `ORCH_TEXT_SHORT=1` **且** state head 开着
#: 时使用（`ORCH_STATE_HEAD=1`，见 `session._do_describe`）。
#:
#: 分工：枚举槽位（ATTENTION / ACTION / EMOTION / BEHAVIOR / RELATION，
#: 以及 ENV 的场景/光线、POSITION 的九宫格）全部交给 state head 的
#: typed-question 模型；这里只留 **E2 表达不了的自由文本**：
#:
#: * `FEATURE` —— 性别/年龄/穿着/配饰（只有文字能描述）
#: * `ENV`     —— 物品与环境要素（E2 里 `ENV.objects` 就是留给文本的）
#: * `POSITION`—— 与物体的相对位置、被指向/拿起的物体（同上）
#: * `DESCRIPTION` —— 一句话概括（**不是** 8 键字段，`parse_state` 会丢掉它，
#:   它只活在原文里进 Agent 的 `content.VLM`）
#:
#: ⚠️ 不要把枚举槽位也一起砍掉却没有 state head —— 那就什么都不剩了，
#: 所以选择逻辑里那两条条件是**与**关系。
SHORT_FULL_INSTRUCTION = (
    "直接输出对当前画面的结构化状态。"
    "不要确认、不要回应、不要以“好的/收到/现在开始”之类的话开头。\n"
    "本轮**只输出下面 4 行**，顺序照抄：\n"
    "FEATURE: 性别、年龄段、穿着（上衣/下装颜色款式、配饰）\n"
    "ENV: 画面中的物品与环境要素（不写场景类型，不写光线）\n"
    "POSITION: 人物与画面中主要物体的相对位置；被指向或拿起的物体\n"
    "DESCRIPTION: 一句话概括画面里正在发生什么（不超过 60 字）\n"
    "不要输出 ATTENTION / ACTION / EMOTION / BEHAVIOR / RELATION —— "
    "那几个字段由另一个模型判定，你再写只会重复。"
)

#: **短文本**版每轮指令。与 `SHORT_FULL_INSTRUCTION` 同集，只是更短。
#:
#: 仍然保留「短暂动作即使已经结束也要报」这类硬要求吗？**不保留** ——
#: 短文本这一侧已经没有 ACTION 了，那条护栏属于 state head 的问句判据
#: （见 `state_head/questions.py` 的 `ACTION_WINDOW_HINT`）。
SHORT_TURN_INSTRUCTION = (
    "本轮只输出当前这些自由文本字段，4 行，顺序照抄：\n"
    "FEATURE: 性别、年龄段、穿着（如「青年女性，灰色短袖」），不写发型细节\n"
    "ENV: 画面中的物品（如「桌上一个白色杯子」）；不写场景类型、不写光线\n"
    "POSITION: 与主要物体的相对位置、被指向或拿起的物体（如「站在展台右侧」）\n"
    "DESCRIPTION: 一句话概括正在发生什么（不超过 60 字）\n"
    "不要输出 ATTENTION / ACTION / EMOTION / BEHAVIOR / RELATION，"
    "也不要写“无变化”这类话；不确定的写“不可见”。"
)


#: **desc-only** 模式的 system prompt —— schema §3 对齐后的默认形态。
#:
#: 分工是**互斥且完备**的：
#:
#: * state head（OneJev）答 schema 里**所有枚举槽位**：
#:   ATTENTION.orientation/gaze、ACTION.events[].type/count、
#:   EMOTION.visual、BEHAVIOR.posture/motion/activity/duration_s、
#:   RELATION.person_count/interaction/target、ENV.scene/lighting、
#:   POSITION.region；
#: * Qwen3-Omni 只答**自由文本槽位**：FEATURE.appearance/accessories、
#:   ENV.objects —— 再加一句 description。
#:   （`POSITION.relative_positions` 原本也在文本侧，2026-11-20 为压延迟
#:   砍掉了这一行；只剩 Jev 的 `POSITION.region`。）
#:
#: ⚠️ 为什么必须换 **system prompt** 而不是逐轮注入：2026-10-28 实测，
#: 注入指令带不动"输出哪些类别"（151 字普通段落），也带不动详细程度
#: （147→151 字，两版都失败）。见本模块顶部那两张表。
#:
#: ⚠️ `ENV` 行只说物品，**不说场景类型与光线** —— 那两个枚举由 state head
#: 给，两边都写只会出现两份对不上的结论。
#:
#: ## 为什么是**两行**（2026-11-20）
#:
#: 三行版每轮约 50 字、~1s；砍掉 POSITION 并压成关键词后实测 ~0.42~0.54s
#: （640px 帧、1s 音频块），仍高于 Jev 的 ~0.2s，但少了近一半。再往下压
#: 的瓶颈是固定开销（一帧视觉编码 + 一轮 decode），不是提示词。
#:
#: **代价**：`POSITION.relative_positions` 不再有文本来源（Jev 只给
#: `POSITION.region`）。需要那半句时把 POSITION 行加回来，或用
#: `ORCH_OMNI_DESCRIBE_SYSTEM_PROMPT` 显式覆盖。
#:
#: ## 模板里**不能出现符号**，示例也**不能贴近常见场景**（2026-11-20 实测）
#:
#: 上一版模板写成 `FEATURE: 人物 —— 性别、年龄段、穿着、配饰`，模型把
#: `——` 当成字段内的固定分隔符照抄，每行 3~4 个，两行白多出 ~18 字：
#: 54 字 / 568ms（中位）vs 无符号模板 37 字 / 501ms。而且「不要标点」
#: 与模板里的破折号自相矛盾 —— 示范的权重比指令大。
#:
#: 另一个坑：示例内容若和画面接近，模型会**一字不差照抄示例**（实测
#: 4/8 拍输出与示例完全相同）。所以示例用与监控场景无关的「老年男性
#: 黑色风衣 / 室外 红色汽车 树木」，并注明「示例内容与画面无关」。
DESC_ONLY_SYSTEM_PROMPT = """你是视频监控/行为分析助手。每轮只用两行关键词描述当前画面，
格式照下面两行（示例内容与画面无关，只照格式）：
FEATURE: 老年男性 黑色风衣
ENV: 室外 红色汽车 树木
词语之间只用一个空格，不要其它任何符号；每行最多 10 个字；只写画面里
实际可见的信息，不确定就省略；不要与用户对话，不要写 ATTENTION / ACTION /
EMOTION / BEHAVIOR / RELATION / POSITION。"""

#: `main.py` 装配用（`ORCH_TEXT_DESC_ONLY=1` 且没有显式覆盖时选它）。
DESC_ONLY_DESCRIBE_SYSTEM_PROMPT = DESC_ONLY_SYSTEM_PROMPT

#: desc-only 的注入指令。**每轮与起点同一套** —— 起点那份也不需要更长。
DESC_ONLY_INSTRUCTION = (
    "本轮直接输出两行，格式照抄，词语之间只用一个空格（示例内容与画面无关）：\n"
    "FEATURE: 老年男性 黑色风衣\n"
    "ENV: 室外 红色汽车 树木\n"
    "从 FEATURE: 开始，不要确认、不要回应。"
)


#: 每个 typed-question 槽位，在文本侧该**省掉**的那半句。
#:
#: 白名单模式下（`ORCH_STATE_SLOTS` 非空，默认就是那 4 个可靠槽位）
#: Qwen3-Omni 仍然输出全量 8 行 —— 2026-10-28 实测「注入指令带不动
#: 输出哪些类别」，所以只能保留 8 行形状，再逐项交代"这一项别写"。
#: 被省掉的项由 state head 给，编排侧按 `partial_keys()` 拼起来。
_SLOT_OMIT = {
    "attn.orientation": "ATTENTION 里的镜头朝向（面向/侧身/背对）",
    "attn.gaze": "ATTENTION 里的视线去向（看镜头/看屏幕/看别处）",
    "act.*": "ACTION 整行（动作由另一个模型判定，不要输出这一行）",
    "emo.visual": "EMOTION 整行（表情情绪由另一个模型判定，不要输出这一行）",
    "beh.posture": "BEHAVIOR 里的姿态（站立/坐）",
    "beh.motion": "BEHAVIOR 里的移动（行走/靠近/远离/转身）",
    "beh.activity": "BEHAVIOR 里正在做的事（用手机/阅读/工作）",
    "beh.duration": "BEHAVIOR 里的持续秒数",
    "rel.person_count": "RELATION 里的人数",
    "rel.interaction": "RELATION 里的互动方式（交谈/面对面/协作）",
    "rel.target": "RELATION 里的主要交互对象",
    "env.scene": "ENV 里的场景类型",
    "env.lighting": "ENV 里的光线",
    "pos.region": "POSITION 里的九宫格位置（画面中央/左侧…）",
}


def partial_instruction(stage: str, slots) -> str:
    """**白名单模式**的指令：全量 8 行，但省掉 E2 已经答了的项。

    与 `SHORT_TURN_INSTRUCTION` 的区别：短文本版要求"只输出 4 行"，而实测
    证明逐轮注入砍不掉类别（模型会照着自己前几轮的风格继续输出 8 行）。
    所以这里**保留 8 行形状**，只在细节上划清边界 —— 即使模型多写了，
    编排侧拼接时也只是重复一遍同样的结论，不会丢信息。

    ``slots`` 为空（E2 全出）时退回普通指令，等价于老行为。
    """
    base = FULL_INSTRUCTION if stage == "full" else TURN_INSTRUCTION
    omit = [_SLOT_OMIT[s] for s in (slots or ()) if s in _SLOT_OMIT]
    if not omit:
        return base
    return (base
            + "\n下面这些项**由另一个模型判定，不要输出**（同一行里其余部分照常输出）："
            + "；".join(omit) + "。")


#: 「无变化」的判据 —— 命中的**不下发**（保住上一份有效描述，见
#: `interaction/downstream.py` 的 `OmniDescription` 分支）。
_NO_CHANGE = (
    "无变化", "没有变化", "无明显变化", "暂无变化", "无显著变化",
    "画面无变化", "没有明显变化", "无",
)


def is_no_change(text: str) -> bool:
    """这句描述是不是"什么都没变"。

    只在**整句就是那句话**时算命中（去掉标点与空白后精确匹配）——
    不能写成 `"无变化" in text`：那样"人物无变化，但桌上多了一个杯子"
    这种**真有信息**的描述会被误丢。
    """
    s = (text or "").strip().strip("。.！!，,、；;：:~～ \t\n")
    return s in _NO_CHANGE


#: **字段值**里的「无变化」—— 命中就**跳过该键**，不合并。
#:
#: ⚠️ 与 `_NO_CHANGE` 的唯一区别：**不含裸「无」**。`ACTION: 无` 是合法值
#: （"没有动作"），把它过滤掉会让 `ACTION` 永远停在上一个动作上。
#:
#: 为什么需要它：2026-10-28 实测（105 影子 138 次描述）有 4 次模型把
#: 「无变化」当成**字段值**写出来（`ATTENTION: 无变化` / `BEHAVIOR: 无变化`），
#: 全部发生在增量轮。以前这些值会被当成正常状态合并进快照，然后**一直**
#: 留在发给 Agent 的载荷里 —— 字段等于废了。
_NO_CHANGE_VALUE = tuple(x for x in _NO_CHANGE if x != "无")


def _is_no_change_value(value: str) -> bool:
    s = (value or "").strip().strip("。.！!，,、；;：:~～ \t\n*")
    return s in _NO_CHANGE_VALUE


# ---------------------------------------------------------------------- #
#  解析 / 渲染
# ---------------------------------------------------------------------- #

#: 「KEY: 内容」，容忍行首 ``-``/``*``/``•``/``#``、粗体 ``**KEY**``、
#: 方括号 ``【KEY】``/``[KEY]``、中英文冒号。键名必须是纯 ASCII 词，
#: 所以 `ATTENTION（交互注意力）` 这种说明行**不会**被误认成字段。
_LINE_WITH_COLON = re.compile(
    r"^\s*(?:[-*•]+\s*)?(?:#{1,6}\s*)?[\*\[【]*\s*([A-Za-z][A-Za-z0-9_]*)\s*[\*\]】]*"
    r"\s*[:：]\s*(.*)$"
)

#: 「【KEY】内容」/「[KEY] 内容」—— 没有冒号的形态。
_LINE_BRACKET = re.compile(
    r"^\s*(?:[-*•]+\s*)?[\*\[【]+\s*([A-Za-z][A-Za-z0-9_]*)\s*[\*\]】]+\s*(.*)$"
)


def parse_state(text: str) -> dict[str, str]:
    """把一段描述解析成 ``{KEY: 值}``。**宽容解析，解析不到就返回空 dict。**

    调用方的约定（见 `interaction/downstream.py`）：
    * 返回值非空 ⇒ 合并进快照；空 ⇒ **回退原文**，一个字节都不丢。

    ## 截断是安全的（但会盖掉一个字段）

    `max_new_tokens` 会在**最后一行**截断（值写到一半）。这里不做"完整性"
    判断，而是靠合并语义兜住：最后一个键的值即使只有半句，也只是盖掉那
    **一个**字段。每轮都是全量、下一轮必然重写全部字段 ⇒ 误差只存活一轮。

    ## 未知键一律丢掉

    模型偶尔会自己造字段（`SCENE:`/`SUMMARY:`），丢了不影响契约；反过来
    放进来会让渲染顺序不稳定。

    ## 值是「无变化」的键也丢掉

    模型会把它当成字段值写出来（实测 4/138）。合并进去会把字段永久污染，
    所以跳过该键、保住旧值 —— 见 `_NO_CHANGE_VALUE`。
    """
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_WITH_COLON.match(line) or _LINE_BRACKET.match(line)
        if m is None:
            continue
        key = m.group(1).upper()
        if key not in _KEY_SET:
            continue
        value = m.group(2).strip().strip("*").strip()
        if not value:
            continue
        if _is_no_change_value(value):
            # 模型把「无变化」当成字段值写了 ⇒ 跳过该键，**保住旧值**。
            # 合并进来会把字段永久污染成「无变化」（见 `_NO_CHANGE_VALUE`）。
            continue
        out[key] = value
    return out


def render_state(state: dict[str, str]) -> str:
    """按 `STATE_KEYS` 的固定顺序渲染成 ``KEY: 值`` 多行文本（缺的键跳过）。"""
    return "\n".join(f"{k}: {state[k]}" for k in STATE_KEYS if state.get(k))


#: 对话式回复的开头（system prompt 没生效时的启发式告警）。
_CONVERSATIONAL_PREFIXES = (
    "您好", "你好", "您好，", "我可以", "请问", "很高兴", "有什么可以帮",
    "需要我", "我是", "当然", "好的，", "嗯，您", "抱歉",
)


def looks_conversational(text: str) -> bool:
    """这句描述**像不像在跟人对话**（而不是在描述画面）。

    ⚠️ 这是个**启发式告警**，不用来丢数据 —— 只用来在日志里提醒
    "prompt 没镇住，模型又开始聊天了"。判错了的代价只是一条日志。
    """
    s = (text or "").lstrip()
    return s.startswith(_CONVERSATIONAL_PREFIXES)
