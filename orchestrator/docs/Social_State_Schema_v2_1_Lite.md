# Social State Schema 2.1（轻量感知版）

> 版本：2.1  
> 变体：**2.1-lite / Omni 音视频融合** —— 与纯视觉实现相比只有 `EMOTION` 的模态口径不同
> （综合表情 + 语音语调 + 说话内容语义），其余 7 个字段一致  
> 载荷：**定义与编码分离** —— 同一套语义有 E0（8 行文本）/ E1（扁平槽位文本）/ E2（JSON）
> 三级编码（§4）。当前用 E1，将来升级到 Omni-Jev / State Head 只切到 E2，**协议不变**  
> 定位：面向实时音视频感知模型的轻量状态输出协议  
> 目标：保留核心社交感知能力，减少重复字段、枚举数量和生成 Token  
> 适用链路：Audio/Video → Omni-Jev → Social State → State Engine / Interaction Policy

---

## 1. 设计原则

1. **保留 8 个顶层字段**：ATTENTION、ACTION、EMOTION、FEATURE、BEHAVIOR、RELATION、ENV、POSITION，兼容现有系统划分。
2. **每个字段只回答一个核心问题**：避免同一语义在多个字段重复表达。
3. **枚举优先、文本补充**：稳定且需要下游消费的状态使用短枚举；长尾细节使用简短文本。
4. **只输出可观察信息**：不确定时明确输出 `unknown`，不强行推断心理状态或社会关系。
5. **时间信息由 State Engine 维护**：模型只报告当前状态和新发生的动作事件，不重复生成持续时长、TTL、历史时间戳。
6. **置信度只绑定枚举槽位**：自由文本不要求模型编造数值置信度。
7. **情绪融合音视频（本变体专属）**：`EMOTION` 综合面部表情、语音语调与**说话内容的文本语义**，
   并保留一个纯视觉旁证槽位 `visual` 用于冲突检测与视觉能力回归（见 §3.3）。
8. **定义与编码分离**：§2、§3 的字段与枚举是**唯一语义定义**；E0 / E1 / E2 只是它的三种载荷形态
   （见 §4）。将来换 Omni-Jev / State Head 只切编码，**字段、枚举、空值语义一行都不改**。

## 2. 八字段精简定义

| 字段 | 中文名称 | 核心问题 | 保留内容 | 明确不做 |
|---|---|---|---|---|
| `ATTENTION` | 交互注意力 | 人在看哪里、朝向哪里？ | 朝向、视线目标 | 不描述姿态和持续动作 |
| `ACTION` | 动作表达（瞬时事件） | 刚刚发生了什么短时动作？ | 点头、摇头、挥手、指向等事件 | 不描述持续运动 |
| `EMOTION` | 情绪表达（音视频融合） | 人物呈现什么情绪？ | 单一主情绪标签 + 纯视觉旁证 | 不推断真实心理；不输出策略结论 |
| `FEATURE` | 自身特征 | 人物外观是什么？ | 衣着、配饰、显著外观 | 不推断身份、职业；年龄只写可观察的年龄段 |
| `BEHAVIOR` | 连续行为（持续状态） | 人当前处于什么姿态或运动状态？ | 姿态、移动状态 | 不重复记录瞬时手势 |
| `RELATION` | 用户关系 / 多人互动 | 场景中有几个人、是否存在互动？ | 人数、互动对象/互动状态 | 不轻易推断同事、亲友、主客等关系 |
| `ENV` | 环境 | 人处于什么环境？ | 场景类型、光照、关键环境物体 | 不包含音频噪声信息 |
| `POSITION` | 位置 | 人在画面哪里、与物体是什么空间关系？ | 画面区域、关键相对位置 | 不输出 bbox、世界坐标和精确距离 |

### 字段边界

- **ATTENTION**：朝向和视线；不等于“是否愿意互动”。
- **ACTION**：短时离散事件，例如挥手、点头；事件发生后可由 State Engine 暂存。
- **EMOTION**：**融合模态** —— 表情 + 语调 + 说话内容语义。说话内容指的是**模型直接从音频里听到的语义**，
  不依赖 ASR 转写结果，因此不会引入额外的 ASR 等待。`visual` 槽位保留纯视觉结论，
  仅用于冲突检测与 SOP08 回归，不参与策略。**音频侧真值（SOP15 的 `emotion`）仍由 ASR 提供**，
  本字段不与它合并，二者是可交叉校验的独立信号。
- **BEHAVIOR**：持续状态，例如坐着、站立、行走、靠近；持续时间由 State Engine 计算。
- **RELATION**：优先记录可观察的多人互动，不把空间相邻直接解释为社会关系。
- **POSITION**：只描述画面空间语义；检测框如有需要，由检测器单独提供，不进入本生成 Schema。
- **在场 / 离场不属于 VLM**：人在不在画面里，由 IC（在场、唇动）与检测侧给出。
  VLM 只回答“画面里看得见什么”，不为“人还在不在”单独输出字段，也不承担策略含义。
- **`other` 的消费口径**：枚举里的 `other` 只表示“有该类现象但不属已知取值”，
  下游**一律按 `unknown` 处理**；不要把它当成一个有效类别去做策略。

---

## 3. 精简字段定义

### 3.1 ATTENTION

回答：人物朝向哪里、视线看向哪里？

```json
{
  "orientation": "facing_camera",
  "gaze": "at_camera"
}
```

| 槽位 | 枚举值 | 中文名称 |
|---|---|---|
| `orientation`（朝向） | `facing_camera` | 面向镜头（正对） |
| | `angled` | 斜对镜头 |
| | `sideways` | 侧身 |
| | `back_to_camera` | 背对镜头 |
| | `unknown` | 无法判断 |
| `gaze`（视线目标） | `at_camera` | 注视镜头 |
| | `at_screen` | 注视屏幕 |
| | `at_person` | 看着某人 |
| | `at_object` | 看着某物 |
| | `wandering` | 视线游移 |
| | `unknown` | 无法判断 |

说明：
- 合并身体朝向与头部朝向的细粒度差异，优先输出整体可见朝向。
- `orientation=facing_camera` 不代表 `gaze=at_camera`。
- 不输出“专注程度”“是否愿意交流”等策略性结论。

### 3.2 ACTION

回答：当前观测窗口内出现了哪些短时动作？

```json
{
  "events": [
    {
      "type": "wave",
      "confidence": 0.91
    }
  ]
}
```

| 槽位 | 中文名称 | 类型 |
|---|---|---|
| `events` | 动作事件列表 | 数组；可见但无动作 `[]`，看不到人 `null` |
| `events[].type` | 动作类型 | 枚举（见下表） |
| `events[].count` | 次数 | 整数；**过渡期由模型给**，State Engine 上线后可省略（见 §5.3） |
| `events[].confidence` | 置信度 | 仅 E2；生成式阶段为 `null` |

`type`（动作类型）允许值：

| 枚举值 | 中文名称 |
|---|---|
| `nod` | 点头 |
| `shake_head` | 摇头 |
| `wave` | 挥手 |
| `point` | 指向 |
| `thumbs_up` | 竖大拇指 |
| `raise_hand` | 举手 |
| `clap` | 鼓掌 |
| `hand_over_object` | 递交物品 |
| `other` | 其他（需在 `description` 里简述） |

说明：
- 没有观察到动作时输出空数组 `[]`；**看不到人**时输出 `null` —— 二者不能混（见 §5.1）。
- 每个事件只保留动作类型、次数和置信度；**不要求**模型生成起止时间、目标 ID、事件描述。
- `count` 在本阶段**必须由模型给**：摇头 2–3 次这类信息只有模型的观测窗口能给，
  跟踪模块尚未接入。State Engine 上线后由引擎校正，字段名不变。
- 1~2 秒的短暂动作，即便解码时已经结束，只要落在观测窗口内就要报告。

### 3.3 EMOTION

回答：人物当前呈现什么情绪？

**本变体（Omni）综合三路证据**：面部表情、语音语调（语速/音量/音高/停顿）、
说话内容的**文本语义**。语义来自模型直接从音频听到的内容，不依赖 ASR 转写，
因此不引入额外的 ASR 等待。

```json
{
  "label": "dissatisfied",
  "confidence": 0.82,
  "visual": "dissatisfied",
  "conflict": true
}
```

| 槽位 | 含义 | 说明 |
|---|---|---|
| `label`（情绪标签） | **综合结论**，下游唯一消费的情绪值 | 融合三路证据；不得输出策略结论 |
| `confidence`（置信度） | `label` 的置信度 | 未校准前允许 `null` |
| `visual`（纯视觉情绪） | **仅看面部表情**的结论 | 只用于冲突检测与 SOP08 视觉回归，不参与策略 |
| `conflict`（音画冲突） | `visual` 与语音侧（语调/语义）方向相反 | 布尔，无冲突为 `false` |

`label` / `visual` 共用同一套词表：

| 枚举值 | 中文名称 | 极性 | 判据要点 |
|---|---|---|---|
| `happy` | 高兴 | 正面 | 明显微笑、笑出声等外放愉快 |
| `satisfied` | 满意 | 正面 | 放松 + 认可，常伴轻微点头、嘴角轻扬 |
| `neutral` | 平静（中性） | 中性 | 观察到中性表情；**不代表内心平静** |
| `confused` | 困惑 | 负面 | 眉间收紧、眉头内侧上挑、疑惑 |
| `dissatisfied` | 不满 | 负面 | 轻中度负面，无明显攻击性 |
| `impatient` | 不耐烦 | 负面 | 明显烦躁、紧锁眉头、目光锐利 |
| `angry` | 愤怒 | 负面 | 强烈愤怒线索、嘴角下压 |
| `sad` | 悲伤 | 负面 | 嘴角下垂、眉部呈悲伤特征 |
| `surprised` | 惊讶 | — | 睁大眼、张嘴；**不计入冲突判定** |
| `other` | 其他 | — | 有明显情绪但不属上述类别，需在 `description` 简述 |
| `unknown` | 无法判断 | — | 证据不足；人脸不可见时必须用它，不能沿用上一轮 |

> 相比 2.1 补回 `satisfied`：SOP08 把「满意」与「高兴」列为**不同**标签，而 2.1 只有 `happy`。
> `satisfied` = 放松 + 认可（常伴随轻微点头、嘴角轻扬）；`happy` = 明显外放的愉快。

#### 融合规则（按证据强度，不做简单投票）

先按极性给词表分组 —— 正面 `happy`/`satisfied`，负面 `confused`/`dissatisfied`/`impatient`/`angry`/`sad`，
中性 `neutral`，`surprised` 视语境不计入冲突判定；「方向相反」= 一侧正面、另一侧负面。

| 情形 | `label` | `visual` | `conflict` |
|---|---|---|---|
| 表情与语音一致 | 该结论 | 同 `label` | `false` |
| 表情明确且非中性，语音方向相反 | **表情结论** | 表情结论 | `true` |
| 表情中性，语音有明确倾向 | 语音结论 | `neutral` | `false` |
| 表情不可见/证据不足，语音有明确倾向 | 语音结论 | `unknown` | `false` |
| 两路都证据不足 | `unknown` | `unknown` | `false` |

规则说明：

- **冲突时表情优先并打标**：非言语通道更难伪装。SOP08_2（皱眉但说「我还是比较满意的」）
  ⇒ `label=dissatisfied`、`visual=dissatisfied`、`conflict=true`。
  是否点破、是否复核由 Interaction Policy 决定，模型本身不给策略。
- **面部不可用时由语音接管**：背对镜头、戴口罩、距离远、画面里没有清晰人脸时，
  纯视觉方案只能大量输出 `unknown`；融合后仍能给出可用情绪 —— 这是本变体最主要的收益。
- **`neutral` 不等于没有情绪**：面部中性 + 语音明显不耐烦 ⇒ `label=impatient`。
- `visual` 只在有清晰人脸时才有意义；人脸不可见时必须 `unknown`，**不能沿用上一轮**。
- 不单独生成情绪描述；确有**冲突**时在全局 `description` 里用一句话点出即可
  （例："皱眉，但称满意"），不要逐槽位复述。
- ⚠️ **`label` 不是心理状态**：它是"可观察的现时表达"，不是"用户真实感受"。
- ⚠️ **不要与 ASR 的 `emotion`（SOP15）合并**：那是纯音频模型的独立信号，
  只在段末（`2pass-offline`）有值；本字段有自己的三路证据，两者可交叉校验，
  但一个不能覆盖另一个。

### 3.4 FEATURE

回答：人物有哪些可见外观特征？

```json
{
  "appearance": "灰色针织上衣、深色长裤",
  "accessories": ["黑框眼镜", "胸前工作证"]
}
```

| 槽位 | 中文名称 | 类型 |
|---|---|---|
| `appearance` | 外观（主要衣着与显著特征） | 自由文本 |
| `accessories` | 配饰 | 自由文本数组；无配饰时 `[]` |

说明：
- 以短文本为主，不把服装颜色、款式、图案拆成多个枚举头。
- `appearance` 描述主要衣着和显著外观；`accessories` 只列明显配饰。
- 无明显配饰时输出 `[]`。
- 同一人物持续跟踪期间，稳定外观可由 State Engine 沿用，不必每轮重复生成。
- SOP09 需要性别 / 年龄段时：**E0 / E1 直接写进 `appearance`**（「青年女性，灰色毛衣」）。
  若将来要结构化，在 E2 增加 `gender` / `age_band` 两个枚举槽位即可，**不影响 E0 / E1**。

### 3.5 BEHAVIOR

回答：人物当前保持什么姿态、处于什么运动状态？

```json
{
  "posture": "sitting",
  "motion": "stationary",
  "activity": "using_phone"
}
```

| 槽位 | 枚举值 | 中文名称 |
|---|---|---|
| `posture`（姿态） | `standing` | 站立 |
| | `sitting` | 坐 |
| | `squatting` | 蹲 |
| | `bending` | 弯腰/俯身 |
| | `lying` | 躺 |
| | `unknown` | 无法判断 |
| `motion`（运动状态） | `stationary` | 静止/保持不动 |
| | `walking` | 行走 |
| | `approaching` | 靠近 |
| | `moving_away` | 远离 |
| | `turning` | 转身 |
| | `moving_sideways` | 横向移动 |
| | `unknown` | 无法判断 |
| `activity`（持续活动） | `using_phone` | 使用手机 |
| | `reading_or_viewing` | 阅读/观看 |
| | `working` | 工作 |
| | `observing_object` | 观察展品或物体 |
| | `other` | 其他 |
| | `unknown` | 无法判断 |

另含一个非枚举槽位：

| 槽位 | 中文名称 | 类型 |
|---|---|---|
| `duration_s` | 当前状态已持续秒数 | 整数秒；**过渡期由模型给**，State Engine 上线后可省略（见 §5.3） |

说明：
- `posture`、`motion`、`activity` 分别表示姿态、运动和可观察的持续活动。
- `activity` 仅在活动明确且持续时输出；短时手势统一放入 ACTION。
- `duration_s` 在本阶段**必须由模型给**：SOP10_2「持续远离 N 秒」依赖它，而跟踪/引擎尚未接入。
  只报**本观测窗口内**的时长，长行为由引擎累加。
- 不输出 `phase`、`duration_ms`、`relative_speed`：阶段划分与速度依赖连续帧，由 State Engine 计算。

### 3.6 RELATION

回答：当前场景有几个人，人物之间是否存在可观察互动？

```json
{
  "person_count": 2,
  "interaction": "talking",
  "target": "another_person"
}
```

| 槽位 | 枚举值 | 中文名称 |
|---|---|---|
| `person_count`（人数） | 整数 或 `null` | 画面可见人数；无法可靠计数时为 `null` |
| `interaction`（互动） | `none` | 无互动 |
| | `talking` | 交谈 |
| | `facing_each_other` | 相互朝向（面对面） |
| | `cooperating` | 协作 |
| | `other` | 其他 |
| | `unknown` | 无法判断 |
| `target`（交互对象） | `another_person` | 另一人 |
| | `camera_or_device` | 镜头/设备 |
| | `object` | 物体 |
| | `none` | 无 |
| | `unknown` | 无法判断 |

说明：
- `person_count` 为当前画面可见人数；无法可靠计数时为 `null`。
- 只记录可观察互动，不默认输出人物间的社会身份关系。
- `facing_each_other` 只表示相互朝向，不等于正在交谈。
- 多人场景若需要 person_id 级关系图，应由跟踪/关系分析模块单独提供，避免让生成模型输出冗长的 pair 列表。

### 3.7 ENV

回答：当前是什么场景，有哪些关键环境要素？

```json
{
  "scene": "office",
  "lighting": "normal",
  "objects": ["办公桌", "显示器", "椅子"]
}
```

| 槽位 | 枚举值 | 中文名称 |
|---|---|---|
| `scene`（场景类型） | `home` | 家居 |
| | `office` | 办公室 |
| | `meeting_room` | 会议室 |
| | `store` | 商店/门店 |
| | `exhibition` | 展厅 |
| | `corridor` | 走廊 |
| | `street` | 街道 |
| | `public_space` | 公共空间/广场 |
| | `nature` | 自然户外 |
| | `vehicle` | 车内 |
| | `other` | 其他 |
| | `unknown` | 无法判断 |
| `lighting`（光照） | `bright` | 明亮 |
| | `normal` | 正常 |
| | `dim` | 偏暗 |
| | `backlit` | 逆光 |
| | `unknown` | 无法判断 |

说明：
- `objects` 为少量关键环境物体的自由文本数组，不罗列背景中的所有物品。
- 删除 `other_person_count`，人数统一由 RELATION 提供。
- 噪声、嘈杂程度、SNR 等归音频侧，不放入视觉 ENV。

### 3.8 POSITION

回答：人物在画面中的哪个区域，与关键物体有什么相对位置关系？

```json
{
  "region": "center",
  "relative_positions": ["位于桌子左侧"]
}
```

| 槽位 | 中文名称 | 类型 |
|---|---|---|
| `region` | 画面区域 | 枚举（见下表） |
| `relative_positions` | 相对位置 | 自由文本数组，如「位于桌子左侧」「站在设备旁边」 |

`region`（画面区域）允许值：

| 枚举值 | 中文名称 |
|---|---|
| `top_left` | 左上 |
| `top_center` | 上中 |
| `top_right` | 右上 |
| `middle_left` | 左中 |
| `center` | 中央 |
| `middle_right` | 右中 |
| `bottom_left` | 左下 |
| `bottom_center` | 下中 |
| `bottom_right` | 右下 |
| `unknown` | 无法判断 |

说明：
- **删除 `bbox_xyxy_norm`。**
- 删除 `camera_distance_appearance`，避免与人物画面占比、区域判断重复，且容易被误解为真实距离。
- `relative_positions` 只保留对交互理解有价值的关系，例如“位于桌子左侧”“站在设备旁边”。
- 精确坐标、检测框和空间定位由专用检测/跟踪模块提供。

---

## 4. 编码分级：一份定义，三种载荷

### 4.0 总则

**§2、§3 的字段与枚举是唯一的语义定义（Semantic Schema）。** §4 定义的是它的**载荷编码
（Wire Encoding）** —— 同一套语义在不同部署阶段用不同形态传输，**不改变字段含义**。

这样做的目的：当前生成式基线（Qwen3-Omni / MiniCPM-o 直接生成文本）要的是**短、稳、好解析**；
将来 Omni-Jev 的 State Head 要的是**带类型的槽位和置信度**。两者不应该是两份协议。

三条硬约束：

1. **编码由部署配置决定，不由模型猜**。同一条链路只跑一种编码。
2. **E2 是超集**：E0 / E1 能无损映射到 E2（补 `null` 置信度即可）；反过来 E2 能降级到 E0 / E1。
3. **枚举 token 只出现在 E1 / E2**；E0 允许自然语言，但其语义必须能落到同一套枚举上（用于离线对齐）。

### 4.1 E0 —— 8 行自然语言文本（当前生产形态）

每个字段一行 `KEY: 值`，值是自然语言短句，**不强制枚举**。这是编排侧现在跑的形态
（`parse_state()` 的正则直接解析，`render_state()` 原样渲染给 Agent）。

```text
ATTENTION: 面向屏幕，注视镜头
ACTION: 无
EMOTION: 平静
FEATURE: 青年女性，灰色毛衣，长裤
BEHAVIOR: 站立 6 秒
RELATION: 单人
ENV: 室内展厅，光线明亮
POSITION: 画面中央
```

- **不输出** `confidence` / `description` / `schema_version` / `count` / `duration` 之外的结构字段。
- 优点：零改动接入现有编排；LLM 可读性最好。
- 缺点：无法结构化消费，离线评测要靠人工/规则对齐。

### 4.2 E1 —— 扁平槽位文本（**当前推荐采用**）

同样是 8 行，但值用固定分隔符承载槽位。**语义与 E2 完全等价**，只是不套 JSON 语法。

```text
ATTENTION: facing_camera|at_camera
ACTION: nod*2
EMOTION: dissatisfied|visual=dissatisfied|conflict
FEATURE: 青年女性，灰色毛衣，长裤|黑框眼镜
BEHAVIOR: standing|stationary||12s
RELATION: 2|talking|another_person
ENV: office|normal|办公桌,显示器
POSITION: center|位于办公桌后方
```

约定：

| 规则 | 说明 |
|---|---|
| 槽位顺序 | 严格按 §3 的槽位顺序，**缺省槽位用空位占位**（如 `BEHAVIOR: standing|stationary||12s` 的 `activity` 为空） |
| 枚举值 | 一律英文 token，与 E2 同一套 |
| 自由文本槽位 | 原样放在对应位置；数组用 `,` 分隔 |
| 多值槽位（`ACTION.events`） | `type*count`，多个用 `;` 分隔；无事件写 `-` |
| 分页 | 本编码**不承载** `confidence` / `description` / `schema_version` |
| 缺键 | 整行缺失 = **沿用上一轮**（见 §5），不是清空 |

**为什么推荐它**：编排侧现在有两条出口 ——
① `parse_state()` 按 `KEY: 值` 解析后 `render_state()` 给 Agent；
② `downstream.py` 的 `_push_vision_to_ic()` 把 8 个键转小写，作为 `apply_vision(**kwargs)`
的**字符串参数**送给 InteractionCore。
E1 的值仍是字符串 ⇒ **两条出口都零改动**，编排侧只多一个 split。改成嵌套 JSON 则两条都要改，
且 JSON 塞进 Agent 的 system prompt 可读性反而更差。

### 4.3 E2 —— JSON（Omni-Jev / State Head 目标形态）

```json
{
  "schema_version": "2.1",
  "description": "用户坐在办公桌前，面向镜头，表情平静；桌上有显示器和水杯。",
  "state": {
    "ATTENTION": {
      "orientation": {"label": "facing_camera", "confidence": null},
      "gaze": {"label": "at_camera", "confidence": null}
    },
    "ACTION": {"events": []},
    "EMOTION": {
      "label": {"label": "neutral", "confidence": null},
      "visual": {"label": "neutral", "confidence": null},
      "conflict": false
    },
    "FEATURE": {
      "appearance": "灰色针织上衣",
      "accessories": ["黑框眼镜"]
    },
    "BEHAVIOR": {
      "posture": {"label": "sitting", "confidence": null},
      "motion": {"label": "stationary", "confidence": null},
      "activity": {"label": "using_phone", "confidence": null}
    },
    "RELATION": {
      "person_count": 1,
      "interaction": {"label": "none", "confidence": null},
      "target": {"label": "camera_or_device", "confidence": null}
    },
    "ENV": {
      "scene": {"label": "office", "confidence": null},
      "lighting": {"label": "normal", "confidence": null},
      "objects": ["办公桌", "显示器", "水杯"]
    },
    "POSITION": {
      "region": {"label": "center", "confidence": null},
      "relative_positions": ["位于办公桌后方"]
    }
  }
}
```

> ⚠️ 上例的 `confidence` 全部是 `null`：**生成式阶段不许自报数字**（见 §6）。
> 只有 State Head 上线后才有真实数值，届时 E2 不变，只是 `null` 变成浮点数。

### 4.4 编码选择

| 阶段 | 编码 | `confidence` | `description` | 每轮字段 |
|---|---|---|---|---|
| 当前：生成式基线接入编排 | **E1**（或 E0 过渡） | 不输出 | 不输出 | 8 键；稳定键可省（见 §5） |
| 离线评测 / 对比实验 | E2 | `null` | 可选 | 8 键全量 |
| Omni-Jev + State Head | E2 | 浮点数（已校准） | 可选 | 8 键全量 + 引擎补时间 |

### 4.5 输出约束（三种编码通用）

- **枚举值一律输出英文 token**（`facing_camera`、`dissatisfied`…）；本文档里的中文名称只用于**说明与提示词释义**，
  不进入模型输出。下游解析、日志检索和跨语言对接都不用做映射。
- 每轮按 §2 的固定顺序输出 8 个字段，减少解析复杂度。
- **`confidence` 只在 E2 出现**，且生成式阶段一律 `null`；不要把它嵌进 E0 / E1 的每一行。
- 多标签事件使用数组；自由文本不附加数值置信度。
- 不确定的枚举统一使用 `unknown`；**不要与 `null` 混用**，语义分工见 §5.1。
- `description` 只在 E2 出现，且仅在需要时输出；E0 / E1 不要为它单独花 token。
- 不输出每字段重复的 `evidence`、`observed_at_ms`、`duration_ms`、`valid_until_ms`、`age_ms`。
- 不输出策略结论，例如“应该主动交流”“应该打断用户”。

## 5. 空值语义、状态沿用与时间

### 5.1 空值语义（**规范，不留“或”**）

2.1 删掉了字段级 `status`，所以“看不见”和“看见了但没有”必须靠值本身区分。
下表的三种空值**分工固定**，三种编码都适用：

| 写法 | 含义 | 用在 |
|---|---|---|
| `unknown` | 枚举槽位证据不足，无法判定 | **只能**用于枚举槽位 |
| `null` | 该槽位 / 字段本回合不可观察 | 枚举槽位可用 `null` 表示“人不在画面”；`person_count`；数组槽位 |
| `[]` | 观察到了，但确实没有（无动作 / 无人 / 无配饰） | 数组槽位 |

要点：

- `ACTION.events = []` ⇒ **人可见、只是没做动作**；`ACTION.events = null` ⇒ **看不到人**。
  这两件事对交互策略完全相反（一个可以继续播报，一个该走离场），不能混。
- `RELATION.person_count = 0` 与 `null` 不同：`0` = 确认画面里没人，`null` = 数不清。
- 枚举槽位**不要**用空串；空串一律视为解析失败。

### 5.2 状态沿用：缺键 ≠ 清空

| 编码 | 整行 / 整键缺失的含义 |
|---|---|
| E0 / E1 | **沿用上一轮**（编排侧 `parse_state` + `update` 天然如此） |
| E2 | 字段缺失 = `null`（未观测），不沿用 |

推论：**E0 / E1 允许省略稳定槽位**，这是当前最划算的一处精简 ——

- 每轮只输出 **6 个动态字段**：`ATTENTION` / `ACTION` / `EMOTION` / `BEHAVIOR` / `RELATION` / `POSITION`；
- `FEATURE` / `ENV` 在会话起点（`full`）给一次，之后省略，由编排侧缓存补齐；
- ⚠️ 唯一例外：**`RELATION.person_count` 变化时必须重出 `FEATURE`**（换人 / 多人进场）。

> 这不是退回“只报变化的增量”。增量方案的失败点是**变化的字段没报**（实测 06_1 的
> `ATTENTION` 一路卡在“面向屏幕”）。`FEATURE` / `ENV` 在单会话内本来就是静态的，缺省是安全的。

### 5.3 时间与次数：模型给，还是引擎给

2.1 的目标是由 State Engine 计算持续时长与事件次数。但**引擎上线前，模型是唯一来源**，
而 SOP07_2（摇头 2–3 次）、SOP10_2（持续远离 N 秒）依赖这两个值。

过渡规则（**字段名和单位全程不变，下游无需区分来源**）：

| 槽位 | 现在（无引擎） | 引擎上线后 |
|---|---|---|
| `ACTION.events[].count` | 模型给整数 | 引擎按事件轨迹校正；模型可省略 |
| `BEHAVIOR.duration_s` | 模型给整数秒 | 引擎按连续帧计算；模型可省略 |

- 单位固定为**秒**，不要输出 `ms`、小数或区间。
- 模型只能估“本观测窗口内”的时长；超过窗口的长行为写当前窗口时长即可，由引擎累加。

### 5.4 其余时间状态由谁维护

- **新观测**：使用本轮模型结果更新状态；观测时间由接收方打（编排侧用 `last_frame_at`，IC 用 `now_ms()`），
  **模型不输出时间戳**。
- **无新帧不描述（协议级约束）**：距最近一次**真正送出**的视频帧超过窗口阈值（编排侧默认 5s）时
  **不触发本轮描述**，保留上一份快照。没有画面时模型不会说“看不到”，而是按字段格式编造一份格式完好的状态
  —— 已在 SOP 片段尾巴上复现两次。
- **短时事件**：`ACTION` 按配置的短 TTL 暂存，过期清除（TTL 由引擎定，模型不输出）。
- **不可见 / 未检测 / 证据不足**：模型只输出 `unknown` / `null`；三者的细分由检测/跟踪模块
  或在场状态（IC）提供，**不要求生成模型区分**。
- **人在不在画面里**：归属 IC（在场 / 唇动）与检测侧，**不是 VLM 的职责**；
  VLM 的 `POSITION = null` 只表示“本回合看不到”，不承担策略含义。

## 6. 置信度使用原则

- 单选枚举：从候选类别 logits 经 softmax 得分。
- 多标签/事件：按标签使用 sigmoid 或事件分数。
- 自由文本：不要求模型自报数值置信度。
- 原始分数不等于真实正确概率；上线前必须使用独立标注集校准，并检查 ECE、Brier Score 或可靠性曲线。
- 在尚未接入可靠 logits / State Head 时，置信度可先输出 `null`；不要让模型凭语言生成一个看似精确的概率。
- **编码级规则**：`confidence` 只属于 E2。E0 / E1 完全不输出它 —— 生成式阶段每轮多 12 个浮点数
  对下游没有任何信息量，只增加 token 与解析负担。State Head 上线后 E2 结构不变，`null` 换成实数。

## 7. 对模型推理效率的建议

1. **固定短 Schema**：将字段定义、枚举白名单放入固定 Prompt 前缀，避免每轮重复传入长说明。
2. **限制文本长度**：`description` 一句话；FEATURE、ENV、POSITION 只输出关键短语。
3. **减少重复字段**：不重复输出 evidence、时间戳、duration、bbox、人数副本。
4. **按槽位输出置信度**：仅在 E2 且已接入 State Head 时输出；E0 / E1 不输出，自由文本永不输出。
5. **状态复用放在模型外**：稳定字段由 State Engine 按 TTL 维护。⚠️ 但**“复用”不等于“少生成变化字段”** ——
   E0 / E1 可以省略 `FEATURE` / `ENV`，**不能**省略 `ATTENTION` / `EMOTION` 等动态字段（增量方案的实测失败点）。
6. **先保证稳定结构**：采用固定字段顺序和短枚举 token，降低解析成本与输出长度。
7. **按编码裁剪载荷**：`confidence` / `description` / `schema_version` / `evidence` / 时间戳一律不进
   E0 / E1 —— 这几项合起来是当前每轮最主要的冗余。
8. **升级不换协议**：换 Omni-Jev / State Head 时只切编码（E1 → E2），字段、枚举、空值语义都不动。

> 说明：Schema 精简能减少输出 Token 和 JSON 解析开销；实际端到端延迟仍取决于输入帧数、音频长度、Prefill、模型解码速度及并发调度。建议用同一组 SOP 视频对比 2.0 与 2.1 的输出 Token、首包延迟、完整解码延迟和状态准确率。

---

## 8. 2.0 → 2.1（含 2.1-lite 变体）主要变更

| 原字段/设计 | 2.1 处理 | 原因 |
|---|---|---|
| ATTENTION 的 body/head orientation、gaze、activities | 合并为 orientation、gaze；持续活动归入 BEHAVIOR.activity | 减少字段交叉，保留可观察的持续活动 |
| ACTION 的 target、start/end 时间 | 删除生成字段 | 由事件检测/State Engine 补充 |
| ACTION 的 `count` | **2.1-lite 过渡期保留**（§5.3） | 跟踪模块尚未接入，SOP07_2「摇头 2–3 次」只有模型的观测窗口能判；引擎上线后改由引擎校正，字段名不变 |
| EMOTION 的长描述与多种近义情绪 | 精简为单一主标签 | 降低分类歧义和输出长度 |
| FEATURE 的颜色、款式、图案多层枚举 | 合并为 appearance 文本 + accessories | 外观组合长尾，不值得多头生成 |
| BEHAVIOR 的 phase、direction、speed | 删除 | 依赖连续帧，由 State Engine 计算 |
| BEHAVIOR 的 `duration` | **2.1-lite 过渡期保留为 `duration_s`**（§5.3） | 引擎未接入时，SOP10_2「持续远离 N 秒」只有模型能给；字段名不变，上线后改由引擎算 |
| RELATION 的 pair 列表、距离、队形、社会关系推断 | 精简为 person_count、interaction、target | 保留核心多人互动信息，避免不可靠推断 |
| ENV 的 other_person_count | 删除 | 与 RELATION.person_count 重复 |
| POSITION 的 bbox_xyxy_norm | **删除** | 由检测器负责，且用户明确不需要该字段 |
| POSITION 的 camera_distance_appearance | 删除 | 与画面区域/人物占比语义重叠 |
| 每个字段重复 status、evidence、observed_at_ms | 移至 State Engine / 外部元数据 | 减少重复 Token |
| 每个自由文本都生成 confidence | 删除 | 避免无校准的伪置信度 |
| **EMOTION 只使用视觉证据** | **2.1-lite：融合表情 + 语调 + 说话内容语义，并保留 `visual` 纯视觉旁证槽位** | Omni 模型能直接听到音频；面部不可见/背对/遮挡时纯视觉只能大量输出 `unknown`。保留 `visual` 是让 SOP08「皱眉却说满意」这类**冲突用例仍可验收**，同时把冲突本身变成下游可用的信号 |
| EMOTION 词表缺 `satisfied` | 2.1-lite 补回 | SOP08 把「满意」与「高兴」列为不同标签 |
| 只有一种 JSON 输出形态 | **2.1-lite 拆成 E0 / E1 / E2 三级编码**（§4） | 当前生成式编排要短、稳、好解析；将来 State Head 要带类型和置信度。两者共用一套语义定义，升级只切编码 |
| 空值语义允许 “`unknown` / `null` / `[]` 由接口层约定” | **2.1-lite 给出固定分工**（§5.1） | `[]`=可见但无、`null`=不可观察、`unknown`=证据不足；`ACTION.events=[]` 与 `null` 对策略的含义相反，不能混 |
| 稳定字段必须每轮全量输出 | **E0 / E1 允许省略 FEATURE / ENV**（§5.2） | 单会话内它们是静态的，缺键=沿用上一轮；动态字段仍必须全量（增量方案实测会 stale carry） |

## 9. 推荐落地方式

三个阶段**共享同一套 §2 / §3 定义**，差异只在编码与置信度来源：

| 阶段 | 编码 | 置信度 | 时间 / 次数 | 改动面 |
|---|---|---|---|---|
| **① 生成式基线（当前）** | E1（可先用 E0 过渡） | 不输出 | 模型给 `count` / `duration_s` | 只改 prompt；编排、IC、Agent 零改动 |
| **② logits / State Head** | E2 | 真实数值（已校准） | 引擎逐步接管 | 加 State Head；Schema 不变 |
| **③ State Engine 接入** | E2 | 沿用② | 全部由引擎计算 | 模型可省略 `count` / `duration_s` / `FEATURE` / `ENV` |

- **第一阶段**：验证 8 字段覆盖率、枚举合法率、输出稳定性、Token 数和端到端延迟。
  预算参考：每轮 ≤150 token 输出、P95 ≤1.2s（按当前 ~150 字 / ~1s 的实测基线）。
- **情绪模态验收（本变体新增）**：用 SOP08_1/08_2 验 `visual` 与 `conflict`
  （08_2 必须 `label=dissatisfied` 且 `conflict=true`）；再用「正对镜头 / 背对镜头 / 戴口罩」
  三组样本验语音接管是否生效（`visual=unknown` 时 `label` 仍应给出非 `unknown` 的结论）。
  与 ASR 的 `emotion`（SOP15）做一次交叉表，确认两者相关但不互相覆盖。
- **第二阶段**：对高价值枚举槽位接入 logits / State Head，优先做 ATTENTION、EMOTION、BEHAVIOR、RELATION。
- **第三阶段**：由 State Engine 负责历史状态合并、TTL、动作事件缓存和持续时间计算。

**升级约束**：① → ② → ③ 之间**不改字段名、枚举取值和空值语义**，只切编码和值的来源。
任何一步如果需要改动 Schema 本身，那是协议变更，必须单独评估。

最终原则：**模型负责看懂当前音视频并输出紧凑状态；State Engine 负责时间连续性；Interaction Policy 负责决策。**
