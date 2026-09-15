# 云端 AEC 在实时语音链路中的有效性问题

**报告日期**：2026-09-15
**涉及服务**：`speech_frontend` 的 `/ws/asr_frontend`（AEC + SE），部署于 `ws://192.168.88.253:30255`
**现象**：实时语音对话中，自己的 TTS 播报被 ASR 识别成"用户说话"（回声未消除）

---

> ## ⚠️ 2026-09-15 后续：根因已定位并修复（调用方侧）
>
> 本报告第 5 节把"11dB 不够"归因于**模型能力**。后续用
> `orchestrator/tests/test_aec_live_fidelity.py` 复测后发现：**离线实验里
> AEC 的输出其实是可用的（人耳听不出回声、ASR 不误识别）**，真正的问题是
> **调用方没能复现离线实验的对齐条件**，导致 AEC 实际等于直通。
>
> 实测（真实服务，扫补偿量 D）：
>
> | 补偿 D | 回声抑制 |
> |---|---|
> | 84ms（真值） | **12.6 dB** |
> | 90ms | 2.5 dB |
> | 250ms（旧默认值） | **0.3 dB** |
>
> 而线上 D 一直是"猜的 250ms"——因为自适应延迟估计路径因一个形状契约
> bug **从未生效**（`estimate()` 要求 mic 与 ref 等长，调用方传的是
> `N` 与 `N+offset`，每次直接 return None）。
>
> **但真机上真正的头号根因不是 D，而是参考轨被清空**：调用方对
> `playback.ended` 也调了 `truncate(from_sample=clock.now())`，而 TTS 是
> **整段一次性**落位、`tts.end` 到达浏览器时**才刚开始播** —— 于是整段
> 参考被清掉，只剩到达那一瞬间的一小截。真机实测：TTS 报 5.85s、写入
> 区间也正好 5.85s，但实际非零只有 **0.6s**（日志 `非零 6/152 = 4%`）。
> 参考只剩 10%，AEC 等于没有 farend。回归护栏见
> `tests/test_ref_truncate_bug.py`。
>
> 修复后真机实测：farend 非零占比 **4% → 43%**、ref 写入区间
> **5.85s → 23.6s**、ERLE **8.3 → 31.6 dB**。
>
> 排查中还试过两个**错误**的假设，均已用实验推翻，记录以免重蹈：
>   · "`RefTrack.place()` 把 TTS 的 int16 量纲原样存进参考轨，farend 比
>     nearend 大 32768 倍是主因" —— 归一化本身是对的（不该靠模型兜底），
>     但 `tests/test_aec_scale_bug.py` 实测模型对量纲**不敏感**：
>     归一化/int16 量纲/放大 100× 的 ERLE 分别为 5.0/5.8/5.9 dB。
>   · "麦克风里没有回声"（由校准通道的 `mic_peak=0.0144` 推出）—— 与
>     "ASR 一直能识别到播报"直接矛盾。校准走的是**完全独立的一条链路**，
>     不能用来推断会话链路。
>
> **因此本报告第 5、7、8 节的结论均已过时**，保留作排查过程记录。
> 当前状态见 `orchestrator/README.md`。

---

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| 回声消除是否生效 | **生效，但最理想也只有 9~11 dB** |
| 是否受延迟影响 | **是**。不预对齐时，>284ms 后抑制降到 1.1 dB |
| 调用方能否适配 | **能**。把 farend 按延迟**延后**对齐后，任意延迟都回到 8.8~11.4 dB |
| 对通话场景是否够用 | **不够**。通话通常需要 ≥20 dB 才能让人听不出回声 |
| 近端语音是否被误伤 | **否**。各场景近端保留 +0.7~+1.5 dB，未见过度抑制 |

**最关键的实测数字**：生产链路的真实条件（延迟 284ms + 预对齐）下，回声抑制仅 **11.4 dB**。

---

## 2. 复现方式

### 2.1 环境

```bash
# AEC 服务需可达
python -c "import socket; socket.create_connection(('192.168.88.253', 30255), 5)"

# 依赖（106 上的 py310 已具备）
pip install websockets numpy
```

### 2.2 一条命令生成全部素材

```bash
cd MiniCPM-o-Demo
python -m orchestrator.tests.gen_aec_report_assets \
    --far  assets/ref_audio/ref_minicpm_signature.wav \   # 模拟 TTS 播放
    --near assets/ref_audio/ref_en_dlc_1.wav \            # 模拟用户说话
    --out  /tmp/aec_report
```

**关键**：`--far` 与 `--near` 必须是**两段不同的语音**。

> ⚠️ **实验设计陷阱**（我们踩过）：若 far 与 near 用同一段音频，
> 则"消掉回声"等于"消掉一切"，会得出"近端被过度抑制 -49dB"的
> **假象**。用不同语音后实测近端保留 +0.7~+1.5 dB，完全正常。

脚本产出 8 个 wav + 1 个 csv（见下节）。

### 2.3 其他实验脚本

| 脚本 | 用途 |
|---|---|
| `orchestrator/tests/test_aec_align.py` | 扫描 AEC 的对齐容忍窗口 |
| `orchestrator/tests/test_aec_prealign.py` | 预对齐对照实验（含 A/B/C 三组） |
| `orchestrator/tests/gen_aec_report_assets.py` | **生成本报告的全部音频素材** |
| `orchestrator/tests/probe_aec.py` | 长时流 / 窗口节奏 / 样本守恒 |

---

## 3. 音频素材（`audio/` 目录）

| 文件 | 内容 | 怎么听 |
|---|---|---|
| `01_near_only.wav` | 近端语音（用户说话），**无回声** | 干净参照 |
| `02_far_reference.wav` | 远端参考（模拟 TTS 播放） | 应该被"消掉"的那路 |
| `03_mic_input.wav` | 麦克风输入 = 回声(延迟284ms, 0.5×) + 近端 | 能同时听到两路 |
| `04_aec_output.wav` | 上者经 AEC 后的输出 | **回声仍在** ← 问题现象 |
| `05_case_D0_input.wav` | 最好情况（延迟=0）的输入 | 对照 |
| `06_case_D0_output.wav` | 最好情况的输出 | 9.2 dB 抑制 |
| `07_prod_input.wav` | **生产链路条件**的输入（284ms） | ← 最重要 |
| `08_prod_aec_output.wav` | **生产链路条件的输出** | **11.4 dB，回声可辨识** |

**建议听法**：依次播 `01` → `03` → `04`。`04` 里应当仍能清楚听到 `02` 的内容
（那就是"被 ASR 识别成用户说话"的那部分）。

---

## 4. 实测数据（`audio/metrics.csv`）

### 4.1 不预对齐

| 真实延迟 D | ERLE(总) | 回声抑制 | 近端保留 |
|---|---|---|---|
| 0 ms（完美对齐） | 4.4 dB | **9.2 dB** | +1.1 dB |
| 100 ms | 3.8 dB | 8.0 dB | +1.4 dB |
| **284 ms（真机实测）** | 3.9 dB | **8.1 dB** | +1.3 dB |
| 500 ms | 0.7 dB | **1.1 dB** | +4.3 dB |

### 4.2 预对齐（把 farend 按 D **延后**）

| 真实延迟 D | ERLE(总) | 回声抑制 | 近端保留 |
|---|---|---|---|
| 0 ms | 4.4 dB | 9.2 dB | +1.1 dB |
| 100 ms | 3.7 dB | 7.6 dB | +1.5 dB |
| **284 ms（真机实测）** | 4.6 dB | **11.4 dB** | +0.7 dB |
| 500 ms | 3.9 dB | 8.8 dB | +1.1 dB |

> **预对齐方向很重要**：因为 `mic[t] = far[t-D]·gain + near[t]`，要让
> 参考与 mic 里的回声分量对齐，必须 `ref[t] = far[t-D]`，即把 far
> **延后** D。（我们最初写成"提前"，得出"预对齐有害"的错误结论。）

### 4.3 术语

- **ERLE(总)**：麦克风总能量被压低多少 dB（含近端，故偏低）
- **回声抑制**：相对**回声分量**的抑制（扣掉近端后的剩余）——**看这个**
- **近端保留**：`输出能量 / 纯近端能量`。0 dB = 近端完好无损

---

## 5. 现象分析

### 5.1 为什么 11.4 dB 不够

11 dB 的抑制意味着回声仍有原始能量的约 **7%**（幅度）。在近端语音
静默时（用户刚说完、TTS 正在播），这 7% 相对信噪比很高，ASR 的 VAD
与识别都能稳定捕捉到 —— 于是播报内容被当成用户输入。

通话级 AEC 通常需要 **20~30 dB**。

### 5.2 回声抑制随延迟的变化

不预对齐时，>284ms 后抑制从 8.1 dB 崩到 1.1 dB —— 说明模型自身
**不做时延估计**，依赖调用方保证对齐。

预对齐后各延迟均稳定在 8~11 dB，说明**对齐是有效的适配手段**，
但它只能把抑制拉回模型的"最好水平"，而这个最好水平本身就只有 9 dB。

### 5.3 与"近端误伤"的关系（澄清）

早期实验曾报告"D<10ms 时近端只剩 -49dB，疑似过度抑制"。
**该结论已作废** —— 那是 far/near 同源的实验设计缺陷所致。
用不同语音复测：近端保留 **+0.7 ~ +1.5 dB**，AEC **不会**吞掉近端语音。

---

## 6. 我们这边的链路现状（已排除的因素）

为排除"是不是调用方接错了"，我们逐项验证过：

| 检查项 | 结果 |
|---|---|
| farend 是否真的送到了 AEC | ✅ 真机日志：非零推送占 32%，峰值 rms=4848 |
| 参考轨时间基准 | ✅ 已用会话采样时钟唯一定位（此前跨时钟域混用是 bug，已修） |
| nearend/farend 成对且等长 | ✅ 协议要求满足 |
| AEC 开关 | ✅ `enable_aec=True`（hello options） |
| SE 是否开启 | ✅ `enable_speech_enhancement=True` |
| 采样率 / 通道数 | ✅ 16kHz 单声道，与 `ASR_FRONTEND_DEFAULTS` 一致 |

**真机实测的声学延迟 D = 4538 采样（284ms）**，其中主要成分是网络传输
与浏览器播放调度，而非纯声学路径。

---

## 7. 希望服务方确认的问题

1. **`/ws/asr_frontend` 是否暴露时延对齐配置？**
   当前 `hello` 的 `options` 只有 `enable_aec` / `enable_speech_enhancement` /
   `nearend_channels` 等开关，未见任何 delay / window / alignment 项。
   若服务端能在内部做对齐，调用方就无需自建。

2. **部署的模型是哪一个？期望的 ERLE 量级是多少？**
   `asr_frontend/config.py` 中 `SD_AEC` 标注 `use_onnx: True`；
   而 `DFSMN_AEC` 分支在代码里是注释掉的。请确认线上实际部署的模型，
   以及它在标准测试集上的 ERLE 指标。

3. **是否存在更强 AEC 模型的部署选项？**
   实测最好 9~11 dB，距通话级（20~30 dB）有较大差距。若库内有其他
   模型或参数档位，希望了解如何切换。

4. **预对齐是否属于预期用法？**
   我们实测"把 farend 按延迟延后"能把 500ms 场景从 1.1 dB 拉回 8.8 dB。
   这是否是推荐做法？还是说服务端本应内部处理？

---

## 8. 我们的临时方案（**已作废** —— 见顶部后续说明）

由于上述问题，**当前已把回声消除改由浏览器原生 AEC 承担**
（`getUserMedia` 的 `echoCancellation/noiseSuppression/autoGainControl`
三项全开），它在设备侧工作、不受网络延迟影响。

云端 AEC 仍保留在链路中（在 284ms 下它相当于直通，不损伤信号），
待服务方确认后可决定是否移除。

> **2026-09-15 更新**：根因不在服务端，而在调用方（见顶部）。上面的
> "直通"判断仍然成立，但**主因是参考轨被 `playback.ended` 清空**（不是
> 模型能力不足）：真机上 5.85s 的播报只有 0.6s 留在了参考轨里，AEC 等于
> 没有 farend。修复后真机 farend 非零占比 4% → 43%、ERLE 8.3 → 31.6 dB。
> D 对齐是**第二位**的因素（容忍窗 ±5ms，D 对齐时抑制 12.6dB）。
>
> **同时注意：本节建议的 `autoGainControl` 全开是错的** —— AGC 是时变
> 增益，会破坏回声路径的线性性，显著拉低云端 AEC 的抑制上限。
> 现在 service 模式下 AGC 已关闭。

---

## 附录：完整复现命令

```bash
# 1. 生成素材（在 106 上）
ssh 192.168.89.106
cd /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
PYTHONIOENCODING=utf-8 /home/dujing/miniconda3/envs/py310/bin/python \
  orchestrator/tests/gen_aec_report_assets.py \
  --far  assets/ref_audio/ref_minicpm_signature.wav \
  --near assets/ref_audio/ref_en_dlc_1.wav \
  --out  /tmp/aec_report

# 2. 对齐窗口扫描
python orchestrator/tests/test_aec_align.py \
  --wav assets/ref_audio/ref_minicpm_signature.wav

# 3. 拉回本地试听
scp -r dujing@192.168.89.106:/tmp/aec_report ./
```

**环境**：Python 3.10 / websockets / numpy；AEC 服务 `ws://192.168.88.253:30255`
