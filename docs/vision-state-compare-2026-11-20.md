# 视觉状态链对比：Qwen3-Omni 全量 8 键 vs Jev 枚举 + Omni 两行文本

**日期**：2026-11-20
**环境**：106（4×RTX 3090），GPU0 上 omni（`-c 200000 --parallel 10`）与 state head（OneJev1.6-0.8B）共卡
**采样**：单路、1fps、640px 长边、每拍随帧送 1s 音频块

---

## 1. 测试视频

| 项 | 值 |
|---|---|
| 路径（106） | `/data/megastore/Projects/DuJing/code/orchdump/s-abdb563e4fb3-raw.mkv` |
| 会话 ID | `s-abdb563e4fb3` |
| 录制时间 | 2026-11-18 18:02 |
| 时长 / 大小 | 61.2 s / 39 MB |
| 视频 | MJPEG 640×360 @25fps |
| 音频 | PCM s16le 16 kHz 单声道 |
| 取用片段 | 第 2 秒起连续 10 拍（每秒 1 帧，即第 2~11 秒） |
| 汇报用短片（前 13 秒，H.264 753 KB） | 106：`orchdump/s-abdb563e4fb3-cmp.mp4`；本地：`D:\work\code\state_cmp\s-abdb563e4fb3-cmp.mp4` |

复现抽帧命令：

```bash
ffmpeg -ss 2 -i s-abdb563e4fb3-raw.mkv -vf "fps=1,scale='min(640,iw)':-2" -frames:v 10 f%03d.jpg
```

---

## 2. 对比方法

两条链**看到的是同一张帧**：omni 侧取 `recent_frames_b64(2)`（已经进了 KV 的帧），原样发给 state head，因此延迟差里不含抽帧/缩放差异。

| 版本 | 组成 |
|---|---|
| **原版** | Qwen3-Omni 每轮吐全量 8 键文本（ATTENTION / ACTION / EMOTION / FEATURE / BEHAVIOR / RELATION / ENV / POSITION） |
| **新版** | Jev（`POST /v1/state`，`stage=full` 全套 16 问）出**全部枚举槽位 + 置信度**；Qwen3-Omni 只出 `FEATURE` / `ENV` **两行关键词**；两条链并行，各自 1Hz 刷新 |

复现命令（仓库根目录）：

```bash
python -m orchestrator.tools.bench_compare <video.mkv> --start 2 --ticks 10 --variant full
python -m orchestrator.tools.bench_compare <video.mkv> --start 2 --ticks 10 --variant desc
```

> 耗时口径：**整轮时间**，含送帧（omni 另含 1s 音频块）与等待解码/JSON 完成。Jev 的数字是 state head 自报的 `latency_ms`。

---

## 3. 结果总览

| 指标 | 原版（Omni 8 键） | 新版（Omni 两行 + Jev） |
|---|---|---|
| 每轮耗时（中位） | **960 ms** | **Omni 495 ms** ／ **Jev 221 ms**（并行） |
| 耗时范围 | 905 ~ 1174 ms | Omni 447 ~ 572 ／ Jev 188 ~ 290 |
| 每轮输出 | 136 ~ 172 字 | Jev：8 键结构化 ／ Omni：34 ~ 43 字 |
| 首拍（冷启动） | 1174 ms | Omni 572 ／ Jev 290 |
| 枚举槽位置信度 | 无 | 有（每槽位 0~1） |

文本侧：**耗时 −48%**（960 → 495 ms），**字数 −77%**（156 → 36 字）。
枚举槽位不再等文本：Jev 约 **220 ms** 返回，独立 1Hz 刷新。

---

## 4. 原版：每轮 8 键全量输出

| 拍 | 耗时 | 字数 | 输出 |
|---|---|---|---|
| 1 | 1174 ms | 158 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 青年女性，浅绿色针织毛衣，深色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内环境，光线偏暗，背景为蓝色发光装饰墙，左侧有绿色应急出口指示牌 \| POSITION: 站在画面中央偏左 |
| 2 | 967 ms | 155 | ATTENTION: 面向屏幕(注视镜头) \| ACTION: 无 \| EMOTION: 不可见 \| FEATURE: 青年女性，浅绿色针织衫，深色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，背景有一块亮着蓝色发光图形的大屏幕和一个绿色的出口指示灯 \| POSITION: 位于屏幕前方 |
| 3 | 932 ms | 136 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 青年女性，米色针织衫 \| BEHAVIOR: 站立，持续 8 秒 \| RELATION: 单人 \| ENV: 室内，背景有发光的电子屏幕，光线偏暗 \| POSITION: 站在屏幕前方 |
| 4 | 1034 ms | 172 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 青年女性，米白色针织上衣，黑色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，背景有发光的蓝色螺旋装置、绿色指示牌、玻璃门和电脑屏幕，光线偏暗，有蓝色冷光 \| POSITION: 位于蓝色发光装置前，面对镜头 |
| 5 | 953 ms | 156 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 青年女性，白色针织上衣，黑色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，背景有发光的蓝色艺术装置、电子屏幕、绿色指示灯 \| POSITION: 位于发光艺术装置与电子屏幕前 |
| 6 | 905 ms | 140 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 青年女性，浅色针织衫 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，背景有发光的蓝色装饰和绿色出口指示灯 \| POSITION: 站在蓝色发光装置前 |
| 7 | 969 ms | 158 | ATTENTION: 面向屏幕(注视镜头) \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 中年女性，米白色针织毛衣，黑色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，背景有发光的蓝色螺旋图案装饰墙，左侧有带绿色箭头的指示牌 \| POSITION: 站在发光背景墙前方 |
| 8 | 994 ms | 163 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 中年女性，穿着米色 V 领毛衣 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内环境，背景有发光的蓝色装饰图案和一个绿色的出口指示灯，光线较暗 \| POSITION: 位于画面中央，前方是发光的蓝色装饰 |
| 9 | 928 ms | 151 | ATTENTION: 面向屏幕(注视镜头) \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 中年女性，灰色针织上衣，黑色长发 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内环境，光线明亮，背景有发光屏幕和绿色安全出口指示牌 \| POSITION: 位于发光屏幕前 |
| 10 | 943 ms | 151 | ATTENTION: 面向屏幕，注视镜头 \| ACTION: 无 \| EMOTION: 平静 \| FEATURE: 中年女性，浅灰色 V 领毛衣 \| BEHAVIOR: 站立，保持不动 \| RELATION: 单人 \| ENV: 室内，光线明亮，背景有发光的蓝色装置、广告牌和绿色出口指示灯 \| POSITION: 站在发光蓝色装置前 |

---

## 5. 新版：Jev（枚举 + 置信度）＋ Omni（两行文本）

| 拍 | Jev | Omni | Jev 输出（节选，括号内为置信度） | Omni 文本 |
|---|---|---|---|---|
| 1 | 223 ms | 563 ms / 40 字 | 面向镜头(0.99) gaze=注视镜头(0.64) 站立(0.98) 静止(0.96) 1人 场景=公共场所(0.26) 光线暗(0.95) 居中(0.92) | FEATURE: 青年女性 黑色长发 \| ENV: 室内 蓝色灯光 绿色出口指示 |
| 2 | 239 ms | 550 ms / 38 字 | 同上，场景=公共场所(0.29) | FEATURE: 青年女性 深色长发 \| ENV: 室内 蓝色灯光 暗色墙壁 |
| 3 | 231 ms | 514 ms / 35 字 | 同上，场景=走廊(0.25) | FEATURE: 中年女性 浅色毛衣 \| ENV: 室内 蓝色屏幕 标识牌 |
| 4 | 290 ms | 572 ms / 43 字 | 同上，场景=公共场所(0.27) | FEATURE: 青年女性 深色头发 灰色毛衣 \| ENV: 室内 蓝色发光装置 绿色标识 |
| 5 | 238 ms | 526 ms / 36 字 | 同上，场景=走廊(0.25) | FEATURE: 青年女性 深色长发 \| ENV: 室内 蓝色灯光 柜子 |
| 6 | 192 ms | 468 ms / 34 字 | 同上，场景=走廊(0.18) | FEATURE: 青年女性 深色长发 \| ENV: 室内 蓝色灯光 走廊 |
| 7 | 192 ms | 463 ms / 34 字 | 同上，场景=公共场所(0.24) | FEATURE: 青年女性 灰色毛衣 \| ENV: 室内 蓝色灯光 屏幕 |
| 8 | 211 ms | 468 ms / 35 字 | 同上，场景=办公室(0.18) | FEATURE: 青年女性 浅色毛衣 \| ENV: 室内 蓝色灯光 显示屏 |
| 9 | 220 ms | 476 ms / 34 字 | 同上，场景=办公室(0.23) | FEATURE: 青年女性 黑色长发 \| ENV: 室内 蓝色背景 墙壁 |
| 10 | 188 ms | 447 ms / 36 字 | 同上，场景=公共场所(0.27) | FEATURE: 中年女性 浅色毛衣 \| ENV: 室内 屏幕 蓝色灯光 |

Jev 输出的完整 8 键（第 1 拍为例）：

```json
{
  "ATTENTION": {
    "orientation": {"label": "facing_camera", "confidence": 0.9916},
    "gaze":        {"label": "at_camera",     "confidence": 0.6427}
  },
  "ACTION": {"events": []},
  "EMOTION": {
    "label":   {"label": "neutral", "confidence": 0.7852},
    "visual":  {"label": "neutral", "confidence": 0.7852},
    "conflict": null
  },
  "FEATURE": {"appearance": null, "accessories": []},
  "BEHAVIOR": {
    "posture":    {"label": "standing",   "confidence": 0.9825},
    "motion":     {"label": "stationary", "confidence": 0.9590},
    "activity":   {"label": "other",      "confidence": 0.4637},
    "duration_s": 1
  },
  "RELATION": {
    "person_count": 1,
    "interaction":  {"label": "none", "confidence": 0.7280},
    "target":       {"label": "none", "confidence": 0.7172}
  },
  "ENV": {
    "scene":    {"label": "public_space", "confidence": 0.2606},
    "lighting": {"label": "dim",          "confidence": 0.9464},
    "objects": []
  },
  "POSITION": {
    "region": {"label": "center", "confidence": 0.9198},
    "relative_positions": []
  }
}
```

`FEATURE` 由 Omni 文本补齐（自由文本槽位只走文本侧），`ENV.objects` 同理；两边在编排里合成一份 8 键快照再发给 Agent。

---

## 6. 结论与已知限制

**收益**

1. **延迟**：文本侧 960 → 495 ms；枚举槽位 221 ms 且独立 1Hz，不再被文本在途阻塞。
2. **输出形态**：原版是一段约 156 字的自然语言，字段靠文本约定、**无置信度**、同一场景下人物的年龄/衣色在拍间反复漂移（青年/中年、浅绿/米色/白色/灰色）；新版行为、关系、场景这类判断由结构化枚举给出，并带置信度，可直接做阈值过滤。
3. **字数**：Omni 侧从 156 字压到 36 字，输出 token 少一个量级。

**限制（汇报时请一并说明）**

1. `ENV.scene` 置信度仅 **0.17~0.30**，且在公共场所/走廊/办公室之间跳动，**暂不建议直接用于业务判断**。
2. `POSITION.relative_positions`（原版那句「站在画面中央偏左」）在新版**没有文本来源**，只剩 Jev 的 `POSITION.region=center`。需要保留就把 POSITION 行加回提示词，代价约 +60 ms。
3. `ACTION` 全程为空（`events: []`），本段视频人物站立不动，属正确结果；动作类槽位的真实准确率需要另用动作片段测。
4. 以上为**单视频、单路、10 拍**的对照，用于说明形态与延迟差异，不代表全量准确率。
