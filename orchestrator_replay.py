#!/usr/bin/env python3
"""编排服务（orchestrator）离线音视频回放客户端。

用 Python 扮演**浏览器**，连 `ws://<host>:<port>/v1/orchestrator`，把预录的
音视频文件按实时节奏喂进去，并处理服务端回来的 TTS / ASR / 人脸状态。

与 `streaming_chat_demo.py` 的区别：那个连的是 OmniLLM 的 gateway
（`/v1/realtime`）；这个连的是**编排服务**，链路是
`mic → AEC → {ASR, OmniLLM}` + `TTS → 回浏览器播放 → 参考轨`。
即它跑的是完整的音频编排链路，而不是直连模型。

## 协议要点（少一条就退化成"能跑但不对"）

  · ``session.start`` 必须带 ``caps: ["playback_anchor"]`` —— 否则服务端
    不等 armed 承诺，参考轨退回**预测落位**（AEC 对不齐）。
  · 每个 mic 块必须带 ``ctx_time``（本块**首采样**在 AudioContext 上的
    时刻，秒）和 ``epoch``。服务端靠它拟合「浏览器时钟 ↔ 会话采样」映射。
  · ``armed`` 必须在**第一块 ``tts.audio`` 到达之后**发，不能在
    ``tts.start`` 时发 —— 后者是在 TTS 合成**之前**发出的，那时算的
    "现在 + 提前量"到音频真到达时早已过期（真机实测会早 340ms）。
  · 采集与播放**共用同一个时钟**（浏览器里是同一个 AudioContext）。
    这里的 `VirtualClock` 就是那一个时钟。

## 它不做什么

⚠️ **不模拟声学回声**。mic 通道就是文件里的干净音频，TTS 播报**不会**被
"回采"进输入通道。所以这个工具能验证：

  · ASR 识别、OmniLLM 回复、TTS 合成与播放调度
  · 打断（barge-in）与参考轨截断
  · 锚点协议本身（``session.stats.anchor_source`` 是否为 ``ack``）

但它**测不了 AEC 的回声消除效果** —— mic 里根本没有回声可消。要测回声
闭环请用 `orchestrator/tests/test_duplex_sim.py`（它合成
`mic = 干净人声 + gain × 喇叭播出[t−D]`）。

## 用法

    # 纯音频（wav，任意采样率，会重采样到 16k）
    python orchestrator_replay.py --audio assets/audio/xxx.wav

    # 音视频（视频抽帧：人脸 320×240、Omni 1280×720）
    python orchestrator_replay.py --video assets/video/turnbased/121.mp4

    # 连远程编排服务
    python orchestrator_replay.py --audio xx.wav --host 192.168.89.106 --port 8100

    # 把服务端回来的 TTS 存下来试听 + 打印每轮指标
    python orchestrator_replay.py --audio xx.wav --save-tts /tmp/tts --verbose

    # 模拟中途插话（在指定秒数触发一次 barge-in）
    python orchestrator_replay.py --audio xx.wav --bargein-at 5,12
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# OpenCV 只在需要**开窗/写视频**时才用得到 —— 纯音频回放不该被它拖累，
# 所以按可选依赖处理，缺失时优雅降级（与 Speaker 的 sounddevice 同款处理）。
try:
    import cv2
except ImportError:
    cv2 = None

# Windows 控制台默认 GBK，打不出 ⚠/✓ 会抛 UnicodeEncodeError 把整个程序带崩
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:  # noqa: BLE001
    pass

SR = 16000              # 会话采样率（全链路统一）
MIC_CHUNK = 1600        # 100ms —— 与服务端 protocol.MIC_CHUNK 一致
TTS_SR = 24000          # 服务端 TTS 输出采样率

# 视频帧节奏。
#
# ⚠️ 视频**必须与音频块解耦**：人脸要 24fps（41.7ms 一帧）比 100ms 的音频
#    块还密，塞在音频循环里最多只能做到 10fps。所以主循环按细粒度 tick 走，
#    音频、人脸、Omni 各自按自己的截止时间触发。
DEFAULT_FACE_FPS = 24.0
DEFAULT_OMNI_FPS = 1.0
FACE_MAX_W, FACE_MAX_H, FACE_Q = 320, 240, 0.5
OMNI_MAX_W, OMNI_MAX_H, OMNI_Q = 1280, 720, 0.7
TICK_S = 0.004          # 主循环粒度（4ms）—— 足够区分 24fps 的帧间隔


def b64f32(x: np.ndarray) -> str:
    """float32 [-1,1] → base64（服务端 decode_audio_b64 按 float32 解析）。"""
    return base64.b64encode(np.ascontiguousarray(x, dtype=np.float32).tobytes()).decode()


def b64bytes(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _has_display() -> bool:
    """当前环境有没有可用的显示器/GUI 会话。

    ⚠️ 这个检查**必须在开窗前**做 —— cv2 的 GUI 后端在无显示器时**不是
    抛异常，而是直接 abort（SIGABRT 把整个进程带走）**，`try/except`
    根本拦不住。`--show` 现在默认开，没这个检查的话在无头机上一跑就 core dump。
    """
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY"))


def screen_size() -> Optional[Tuple[int, int]]:
    """屏幕像素尺寸；取不到返回 None。

    ⚠️ 为什么需要它：cv2 默认用 ``WINDOW_AUTOSIZE``，窗口**不可缩放**且
    恰好等于图像尺寸 —— 1440×1080 的视频在 1080p 屏上会被任务栏切掉底部
    （字幕正好在底部，就看不到了）。所以要么按屏幕缩放，要么换
    ``WINDOW_NORMAL``；这里两个都做。
    """
    try:                      # 跨平台首选
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        s = (int(r.winfo_screenwidth()), int(r.winfo_screenheight()))
        r.destroy()
        return s
    except Exception:  # noqa: BLE001 —— 无 GUI / 无 tkinter
        pass
    try:                      # Windows 兜底
        import ctypes
        u = ctypes.windll.user32                    # type: ignore[attr-defined]
        return (int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1)))
    except Exception:  # noqa: BLE001
        return None


# ============================================================================
# 虚拟 AudioContext 时钟
# ============================================================================

class VirtualClock:
    """模拟浏览器 ``AudioContext.currentTime``。

    ⚠️ **采集与播放必须共用这一个时钟**。浏览器里两者本来就在同一个
    AudioContext 上，所以 mic 块的 ``ctx_time`` 与播放器的 ``start_ctx``
    天然同源；离线下如果各用各的，服务端拟合出的映射就是错的。

    起点取一个非零值（默认 1000.0）有两个原因：
      ① 服务端把 ``ctx_time <= 0`` 当作"本次没上报"的哨兵值丢弃；
      ② 真实 AudioContext 的 currentTime 本来就是个较大的数。
    """

    def __init__(self, base: float = 1000.0) -> None:
        self._base = float(base)
        self._t0 = time.monotonic()

    def now(self) -> float:
        return self._base + (time.monotonic() - self._t0)


# ============================================================================
# 虚拟播放器（复刻 PcmPlayer 语义）
# ============================================================================

class Speaker:
    """把收到的 TTS 真正**放出来**（可选）。

    没装 sounddevice 或没有声卡时降级为静默，不影响回放流程 —— 回放的价值
    在于协议与指标，出声只是方便人耳确认。

    ⚠️ 出声之后 mic 侧**不会**自动把声音收回去：本工具不模拟声学回声
    （见模块 docstring），所以外放不会污染输入通道。这与真机不同 ——
    真机上喇叭一响，麦克风就收得到，那正是 AEC 要处理的问题。
    """

    def __init__(self, device: Optional[str] = None,
                 sample_rate: int = TTS_SR) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.ok = False
        self.err = ""
        self.dropped = 0
        self._stream = None
        self._sd = None
        # 后台消费队列：`write` 阻塞（按实时速度），绝不能放在事件循环里
        self._q: "queue.Queue" = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._worker_t: Optional[threading.Thread] = None
        try:
            import sounddevice as sd
            self._sd = sd
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32",
                device=device, blocksize=0)
            self._stream.start()
            self._worker_t = threading.Thread(target=self._worker, daemon=True)
            self._worker_t.start()
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            self.err = f"{type(exc).__name__}: {exc}"

    def play(self, pcm24: np.ndarray) -> None:
        """**非阻塞**投递一块 24k int16 PCM 去播放。

        ⚠️ **绝不能在事件循环里同步 `stream.write`**。它是**阻塞**的，
        按实时速度消耗音频 —— 36s 的回复会把 asyncio 事件循环堵满 36s，
        后果是：
          · 时钟 / 所有异步任务被拖慢（TTS 听感"延迟非常大"）
          · 接收循环收不到后续的 tts.* 事件，"剩余时长"卡死不动
          · 收尾判断误判 → **没播完就关连接**
        所以这里只入队，真正的 write 交给后台线程按自己的节奏消费。
        """
        if not self.ok:
            return
        try:
            self._q.put_nowait(np.ascontiguousarray(
                pcm24, dtype=np.int16).astype(np.float32) / 32768.0)
        except queue.Full:
            self.dropped += 1
            # 队列满 = 播放追不上，丢最旧的（新的更重要）
            try:
                self._q.get_nowait()
                self._q.put_nowait(np.ascontiguousarray(
                    pcm24, dtype=np.int16).astype(np.float32) / 32768.0)
            except Exception:  # noqa: BLE001
                pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                x = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if x is None:
                break
            if self._stream is None:
                break
            try:
                self._stream.write(x)
            except Exception as exc:  # noqa: BLE001
                self.err = f"{type(exc).__name__}: {exc}"
                self.ok = False
                break

    def pending_s(self) -> float:
        """还有多少秒的音频**没放完**（队列里排着的）。"""
        try:
            n = self._q.qsize()
        except Exception:  # noqa: BLE001
            return 0.0
        return n * 0.5            # 每块 0.5s（服务端 TTS 的分块大小）

    def stop(self) -> None:
        """打断：丢掉还没播出去的缓冲。"""
        if not self.ok or self._stream is None:
            return
        # 清空待播队列 —— 打断后旧句的剩余部分不该再响
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            self._stream.abort()      # abort 会丢弃缓冲（stop 是播完再停）
            self._stream.start()
        except Exception:  # noqa: BLE001
            self.ok = False

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_t is not None:
            self._worker_t.join(timeout=3.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None


# ============================================================================
# 人脸叠加层
# ============================================================================

# drawFace() 的尺寸是 **CSS 像素**，而它所在的 canvas 宽约 380 CSS px、
# 对着一帧 320 宽的源图。换算到"源帧单位"要除以这个系数。
# 所有叠加层几何都先按源帧算、再乘 k（k = 输出帧宽 / src_w），
# 这样画在 320×180 上和画在 1920×1080 上比例一致。
OVERLAY_UI_SCALE = 380.0 / 320.0
OVL_FONT_CSS = 13.0
OVL_PAD_CSS = 5.0
OVL_LABEL_H_CSS = 19.0
OVL_LABEL_GAP_CSS = 20.0     # 标签底边在框顶上方 20 CSS px
OVL_TEXT_DX_CSS = 5.0

# 中文字体候选（按顺序找第一个存在的）。106 上没有中文字体会走 ASCII 回退。
_CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",          # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",        # Windows 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
)


class FaceOverlay:
    """把人脸框/唇动状态画到视频帧上（复刻 `drawFace()` 的语义）。

    ⚠️ **坐标系是本模块最容易搞错的地方**，先说清楚：

    服务端**从不重采样**我们发去的人脸帧 —— 它把 JPEG 直接喂给 G1，再从
    **同一份字节**里读回尺寸（`face/local_provider.py` 的 `_frame_size`），
    作为 `src_w`/`src_h` 下发。所以 ``box`` 就在**我们发出去的那张图**的
    像素坐标系里，而不是原始视频的。

    映射是**纯比例**、没有 letterbox 偏移（ffmpeg 的 decrease 只缩放不补边）：
        ``x = l · W / src_w``，``y = t · H / src_h``

    ⚠️ 因此缩放系数要**从 src_w/src_h 读**，不能自己按 320×240 重算：
    ffmpeg 的 `force_original_aspect_ratio=decrease` 在源比 320×240 **小**时
    会**放大**（160×120 → 320×240），而浏览器 `grab()` 用 `Math.min(1,…)`
    不放大。两者在这点上不一致 —— `src_w` 才是唯一可信的坐标系。
    """

    def __init__(self, font_path: str = "", scale: float = 1.0,
                 force_ascii: bool = False) -> None:
        self.scale = scale
        self.font = None
        self.ascii_only = force_ascii
        self.hint_shown = False
        if force_ascii:
            return
        try:
            from PIL import ImageFont
        except ImportError:
            self.ascii_only = True
            return
        cands = ([font_path] if font_path else []) + list(_CJK_FONT_CANDIDATES)
        for p in cands:
            if p and os.path.exists(p):
                try:
                    self.font = ImageFont.truetype(p, 16)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if self.font is None:
            self.ascii_only = True      # 有 Pillow 但找不到中文字体
            return
        # 缓存字体度量：逐帧测量也要几毫秒，而标签高度是**常量**
        try:
            bb = self.font.getbbox("测Ag")
            self._font_h = int(bb[3] - bb[1]) + 6
        except Exception:  # noqa: BLE001
            self._font_h = 22
        self._lab_cache: Dict[str, int] = {}

    def _font_w(self, s: str) -> int:
        """量一行文本宽度（带缓存 —— 标签种类就那几种，逐帧测量没必要）。"""
        w = self._lab_cache.get(s)
        if w is None:
            try:
                bb = self.font.getbbox(s)
                w = int(bb[2] - bb[0]) + 6
            except Exception:  # noqa: BLE001
                w = len(s) * 14
            self._lab_cache[s] = w
        return w

    @property
    def ok(self) -> bool:
        return cv2 is not None

    # ------------------------------------------------------------------ #

    @staticmethod
    def resolve_src(msg: dict, fallback: Optional[Tuple[int, int]]
                    ) -> Optional[Tuple[int, int]]:
        """``box`` 所在坐标系的尺寸。优先服务端下发的 src_w/src_h。"""
        tr = (msg.get("tracks") or [None])[0] if msg else None
        if tr and tr.get("src_w") and tr.get("src_h"):
            return int(tr["src_w"]), int(tr["src_h"])
        return fallback

    @staticmethod
    def label_parts(msg: dict) -> List[str]:
        """逐条复刻 drawFace() 的标签拼接（分隔符 ' · '）。

        注意 ``enrolled and name`` 是**与**关系：已注册但 name 为空时落到
        uid 分支、显示"未注册"（与 JS 一致，别想当然改成 or）。
        """
        tr = (msg.get("tracks") or [None])[0] or {}
        ident = msg.get("identity") or None
        parts = ["%.2f" % float(tr.get("score") or 0.0)]
        parts.append("说话中" if tr.get("speaking") else (tr.get("lip") or "SILENT"))
        if ident and ident.get("enrolled") and ident.get("name"):
            parts.append(str(ident["name"]))
        elif ident and ident.get("uid"):
            parts.append("未注册")
        if tr.get("interacting"):
            parts.append("✓唤醒")
        return parts

    # ---- 彩色常量（JS 里是 RGB 十六进制，OpenCV 要 BGR）----
    _BGR_WAKE = (94, 197, 34)      # #22c55e
    _BGR_IDLE = (8, 179, 234)      # #eab308
    _BGR_TXT_WAKE = (172, 239, 134)  # #86efac
    _BGR_TXT_IDLE = (138, 230, 253)  # #fde68a
    _ASCII_MAP = {"说话中": "SPEAKING", "未注册": "UNK", "✓唤醒": "WAKE"}

    def _text(self, img, s: str, org, color):
        """画一行文字。有 Pillow + 中文字体走高质量路径，否则 ASCII 回退。

        ⚠️ **只在标签那一小块上做 Pillow 往返**，不要整帧转。
        实测：对 1440×1080 整帧做 ``Image.fromarray`` + ``np.asarray``
        要 **31ms**，而 24fps 的预算是 41.7ms —— 光这一步就吃掉 3/4，
        再叠上解码和 x264 编码必然掉帧（实测丢 18/220）。
        标签区域只有几百像素宽，裁出来转换快到可以忽略。
        """
        if not self.ascii_only and self.font is not None:
            from PIL import Image, ImageDraw
            th = self._font_h
            tw = self._font_w(s)
            x0, y0 = int(org[0]), int(org[1]) - th
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img.shape[1], x0 + tw + 4), min(img.shape[0], y0 + th + 4)
            if x1 <= x0 or y1 <= y0:
                return
            roi = img[y0:y1, x0:x1]
            pil = Image.fromarray(roi[:, :, ::-1])       # BGR→RGB（仅小块）
            ImageDraw.Draw(pil).text((int(org[0]) - x0, int(org[1]) - th - y0),
                                     s, font=self.font, fill=color[::-1])
            roi[:] = np.asarray(pil)[:, :, ::-1]         # RGB→BGR
            return
        # ⚠️ cv2.putText 用的是 Hershey 字体，**只有 ASCII 32~126** ——
        #    中文和 '·' 会变成 '?'，所以必须转写而不是硬画。
        for k, v in self._ASCII_MAP.items():
            s = s.replace(k, v)
        s = s.replace(" · ", " | ")
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                    cv2.LINE_AA)          # 先描黑边，浅色背景上也看得清
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)

    def draw(self, frame, msg: Optional[dict],
             src_wh: Optional[Tuple[int, int]]) -> None:
        """就地把人脸框与标签画到 ``frame``（BGR）。没有可画的内容就静默跳过。"""
        if not self.ok or not msg:
            return
        tr = (msg.get("tracks") or [None])[0]
        # 与 JS 的 `if (t && t.valid && t.box)` 一致：无效/无框就不画
        if not tr or not tr.get("valid") or not tr.get("box"):
            return
        wh = self.resolve_src(msg, src_wh)
        if not wh or not wh[0] or not wh[1]:
            if not self.hint_shown:
                self.hint_shown = True
                print("  ⚠️ 人脸框坐标系未知（服务端未下发 src_w）—— 跳过叠加")
            return

        H, W = frame.shape[:2]
        sw, sh = wh
        sx, sy = W / float(sw), H / float(sh)
        if abs(sx - sy) > 0.01 * sx:
            # 服务端只在**第一帧**缓存尺寸。这里不一致说明我们中途换了发送
            # 分辨率，框会被拉歪 —— 大声说出来，别默默画错。
            print(f"  ⚠️ 框坐标系与输出帧宽高比不一致（{sw}×{sh} → {W}×{H}）"
                  f"—— 框会被拉歪")

        l, tp, r, b = [float(v) for v in tr["box"]]
        x, y = int(round(l * sx)), int(round(tp * sy))
        bw, bh = int(round((r - l) * sx)), int(round((b - tp) * sy))

        wake = bool(tr.get("interacting"))
        k = (W / float(sw)) / OVERLAY_UI_SCALE * self.scale
        cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                      self._BGR_WAKE if wake else self._BGR_IDLE,
                      max(1, int(round((3 if wake else 2) * k))))

        label = " · ".join(self.label_parts(msg))
        fam = max(1, int(round(6 * k)))          # 与字体大小绑定的缩放因子
        fs = 0.45 * k
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, fam)
        ly = max(int(round(OVL_LABEL_GAP_CSS * k)), th + 4)
        pad = int(round(OVL_PAD_CSS * k))
        x0, y0 = x, max(0, y - ly)
        # 标签底：JS 用 rgba(0,0,0,0.65) 叠在黑底上 ⇒ 等价于把该区域乘 0.35
        x1, y1 = min(W, x0 + tw + pad * 2), min(H, y0 + max(
            int(round(OVL_LABEL_H_CSS * k)), th + pad))
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            roi[:] = (roi.astype(np.float32) * 0.35).astype(np.uint8)
        self._text(frame, label,
                   (x0 + int(round(OVL_TEXT_DX_CSS * k)), y1 - pad),
                   self._BGR_TXT_WAKE if wake else self._BGR_TXT_IDLE)


class FaceWindow:
    """回放时开一个窗口显示视频 + 人脸框/唇动；可选同时录成 mp4。

    ⚠️ **必须跑在独立线程里**。主循环是**实时**节奏（24fps 只有 41.7ms
    预算，还要发 100ms 的音频块），而 1080p 单帧的解码+绘制+编码要
    10~20ms；塞进主循环必然掉帧，进而把音频块也发晚 —— 参考轨落位随之
    偏移，AEC 的结论就不可信了（这个工具测的是协议与对齐，不是画质）。

    帧的来源用**与 `extract_frames()` 同一条 ffmpeg 命令**（只去掉
    `scale`），而不是 cv2.VideoCapture：两者的选帧规则不同（fps 滤镜按
    时间戳、VideoCapture 按位置/关键帧），选出来的可能不是同一张 ——
    那框就会和画面内容整体错开（静物看不出来，转头时很明显）。

    无显示器 / 缺依赖时降级为"只写文件"，两者都没有就整个跳过。
    """

    def __init__(self, video_path: str, out_path: str, fps: float,
                 size: Optional[Tuple[int, int]] = None,
                 show: bool = True, overlay: Optional["FaceOverlay"] = None,
                 crf: int = 20, preset: str = "veryfast",
                 play_audio: bool = False, mux_audio: bool = True,
                 play_speed: float = 1.0,
                 status: "Optional[StatusOverlay]" = None,
                 disp_scale: float = 0.0) -> None:
        self.video_path = video_path
        self.out_path = out_path
        self.fps = max(1.0, fps)
        self.show = show
        self.overlay = overlay
        self.crf = crf
        self.preset = preset
        self.play_audio = play_audio        # 把**原片音轨**也放出来
        self.mux_audio = mux_audio          # 把原片音轨混进输出的 mp4
        self.play_speed = play_speed
        self.status = status
        # 显示缩放系数。0 = 自动（按屏幕可用高度算，并留出任务栏/标题栏）；
        # >0 用指定值。**只影响窗口显示**，写进 mp4 的仍是全分辨率。
        self.disp_scale = disp_scale
        self._scale = 1.0
        # 渲染线程只投帧，**GUI 调用留给主线程**（见 render()）
        self._latest = None
        self.ok = False
        self.err = ""
        self.frames = 0
        self.dropped = 0
        self.size = size or (0, 0)

        self._q: "queue.Queue" = queue.Queue(maxsize=8)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc = None
        self._reader = None
        self._speaker: Optional[Speaker] = None
        self._src_audio: Optional[np.ndarray] = None
        self._win = "orchestrator-replay"

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if cv2 is None:
            self.err = "缺 opencv-python（pip install opencv-python）"
            return
        try:
            self._reader = self._open_reader()
        except Exception as exc:  # noqa: BLE001
            self.err = f"读视频失败：{exc}"
            return
        if self.out_path:
            try:
                self._proc = self._open_writer()
            except Exception as exc:  # noqa: BLE001
                self.err = f"启动编码器失败：{exc}"
                self._reader.kill()
                return
        if self.show and self.play_audio:
            # 原片音轨（16k 单声道 float32）—— 与回放用的是同一份音频，
            # 所以"看到的画面"和"听到的声音"天然同步。
            try:
                self._src_audio = self._extract_audio()
                self._speaker = Speaker(sample_rate=SR)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ 原片声音不可用：{exc}")
                self._speaker = None
            if self._speaker is not None and not self._speaker.ok:
                print(f"  ⚠️ 原片声音不可用：{self._speaker.err}")
                self._speaker = None
        if self.show:
            # ⚠️ 必须是 WINDOW_NORMAL。cv2 默认的 WINDOW_AUTOSIZE **不可缩放**
            #    且窗口尺寸恰好等于图像 —— 1440×1080 在 1080p 屏上会被任务栏
            #    切掉底部，而字幕正好在底部（用户实测看不到）。
            try:
                cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
                self._scale = self._compute_scale()
                if self._scale < 1.0:
                    print(f"  窗口缩放 {self._scale:.2f}×"
                          f"（窗口可拖动/缩放；按 f 全尺寸，ESC 关闭）")
            except Exception as exc:  # noqa: BLE001 —— 无显示器
                print(f"  ⚠️ 开窗失败（{type(exc).__name__}: {exc}）—— 继续但无窗口")
                self.show = False
        self.ok = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _compute_scale(self) -> float:
        """按屏幕可用高度算显示缩放（0=自动时）。

        ⚠️ 不能用整屏高度：标题栏 + 任务栏会吃掉 ~120px，用整屏算出来的
        "刚好装满"实际会被切掉一截 —— 底部字幕就看不到了。
        """
        if self.disp_scale > 0:
            return self.disp_scale
        scr = screen_size()
        if not scr or not self.size[0] or not self.size[1]:
            return 1.0
        avail_h = scr[1] - 120          # 标题栏 + 任务栏余量
        avail_w = scr[0] - 80           # 桌面边距
        return max(0.2, min(1.0, avail_h / self.size[1], avail_w / self.size[0]))

    def _open_reader(self):
        """开 ffmpeg 读原始帧（与抽帧同一条命令，只去掉 scale）。"""
        w, h = self._probe_size()
        cmd = ["ffmpeg", "-loglevel", "error", "-i", self.video_path,
               "-vf", f"fps={self.fps}", "-an", "-sn",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        self.size = (w, h)
        return {"proc": p, "w": w, "h": h}

    def _probe_size(self) -> Tuple[int, int]:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             self.video_path],
            capture_output=True, text=True, check=True)
        w, h = r.stdout.strip().split(",")[:2]
        return int(w), int(h)

    def _extract_audio(self) -> np.ndarray:
        """抽原片音轨 → 16k 单声道 float32。无音轨则返回空数组（不报错）。"""
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", self.video_path, "-vn",
             "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1",
             "-f", "f32le", "pipe:1"],
            capture_output=True)
        if r.returncode != 0 or not r.stdout:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(r.stdout, dtype=np.float32).copy()

    def _open_writer(self):
        """ffmpeg 编码 mp4。

        用 ffmpeg 而不是 cv2.VideoWriter：后者配 mp4v 出来的文件大 5~10 倍，
        且不少播放器直接不认。libx264 + yuv420p + faststart 才是能直接发给
        别人的文件。
        """
        cmd = ["ffmpeg", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(self.fps),
               "-i", "pipe:0"]
        # ⚠️ 把**原片音轨**一起 mux 进来，否则录出来的 mp4 是无声的。
        #    只在 1.0 倍速时接：加速回放时画面被压缩而音轨没被压，两者会
        #    越走越偏 —— 一条**静默错位**的音轨比没有音轨更坑（看的人会
        #    以为声音就是那样的）。
        if self.mux_audio and abs(self.play_speed - 1.0) < 1e-6:
            cmd += ["-i", self.video_path, "-map", "0:v", "-map", "1:a?",
                    "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                self.out_path]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------ #

    def push(self, msg: Optional[dict], src_wh: Optional[Tuple[int, int]]
             ) -> None:
        """主循环调用：把"发这一帧时的人脸状态"入队。

        ⚠️ 只放**元组**不放图像：1080p 一帧 6MB，攒几帧就把内存吃光。
        帧由工作线程自己按序去读，所以这里给的是"当时的状态"而不是
        "绘制时的最新状态" —— 后者会随队列深度静默地引入额外滞后。

        ⚠️ 用 put_nowait：主循环**绝不能阻塞**。落后就丢并计数（丢在入队
        端而不是出队端，出队端丢会让输出视频的时间轴被压缩）。
        """
        if not self.ok:
            return
        try:
            self._q.put_nowait((msg, src_wh))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        n = 0
        while True:
            try:
                msg, src_wh = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if msg is None and src_wh is None and self._stop.is_set():
                break
            frame = self._read_frame(n)
            if frame is None:
                continue
            if self.overlay is not None:
                self.overlay.draw(frame, msg, src_wh)
            if self._proc is not None:
                try:
                    self._proc.stdin.write(
                        np.ascontiguousarray(frame).tobytes())
                except Exception:  # noqa: BLE001 —— 编码器挂了就别再写
                    self._proc = None
            if self.show:
                # ⚠️ 字幕/状态栏只画在**窗口这一份**上，不写进 mp4 ——
                #    它们是服务端实时回的、与视频时间轴不严格对齐，
                #    烧进文件会误导（也会把文件撑大）。
                if self.status is not None:
                    self.status.draw(frame)
                # ⚠️ 缩放**只作用于显示**：写进 mp4 的仍是全分辨率。
                #    这里只做 resize 并把帧投进队列，**绝不碰 GUI** ——
                #    `cv2.imshow/waitKey` 必须在主线程（见 `_render`）。
                if self._scale < 0.999:
                    frame = cv2.resize(frame, None, fx=self._scale,
                                       fy=self._scale,
                                       interpolation=cv2.INTER_AREA)
                self._latest = frame
            # 原片声音：按帧节奏喂对应时长的音频。
            # ⚠️ 必须在**这个线程**里按帧驱动，不能一次性灌进去 ——
            #    一次性写会让声音跑在画面前面（缓冲区攒满就停），
            #    与"边播边对齐"的意图矛盾。
            if self._speaker is not None and self._src_audio is not None:
                per = int(SR / self.fps)
                a = (n * per) % max(1, len(self._src_audio))
                seg = self._src_audio[a:a + per]
                if seg.size:
                    self._speaker.play((seg * 32767.0).astype(np.int16))

            self.frames += 1
            n += 1

    def _read_frame(self, n: int):
        """按序读第 n 帧。视频放完会**绕回**（与服务端发帧的行为一致）。"""
        r = self._reader
        if r is None:
            return None
        need = r["w"] * r["h"] * 3
        buf = b""
        while len(buf) < need:
            chunk = r["proc"].stdout.read(need - len(buf))
            if not chunk:                    # EOF → 重开，从头再读
                r["proc"].stdout.close()
                r["proc"].kill()
                try:
                    self._reader = self._open_reader()
                except Exception:  # noqa: BLE001
                    return None
                return self._read_frame(0)
            buf += chunk
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            r["h"], r["w"], 3).copy()

    def render(self, timeout_ms: int = 1) -> bool:
        """**主线程**调用：显示最新一帧并处理按键。返回 False 表示用户关了窗。

        ⚠️ `cv2.imshow` / `cv2.waitKey` **必须在主线程**。早先把它们放在
        渲染线程里，窗口**完全没有画面**（cv2 高版本对 GUI 事件循环的线程
        有要求，从工作线程调会静默失效 —— 不报错、就是不刷新）。
        所以这里拆成两半：工作线程只负责解码/叠加/编码并投帧，
        GUI 全部由主线程的 `render()` 驱动。
        """
        if not self.show or self._latest is None:
            return True
        try:
            cv2.imshow(self._win, self._latest)
            k = cv2.waitKey(timeout_ms) & 0xFF
        except Exception:  # noqa: BLE001 —— 无显示器
            self.show = False
            return True
        if k == 27:                       # ESC
            self.show = False
            return False
        if k in (ord("f"), ord("F")):
            # 在"适应屏幕"与"1:1 全尺寸"间切换（想看细节时不用改命令行）
            self._scale = 1.0 if self._scale < 0.999 else self._compute_scale()
            cv2.resizeWindow(self._win, int(self.size[0] * self._scale),
                             int(self.size[1] * self._scale))
        return True

    def close(self) -> None:
        if not self.ok:
            return
        self._stop.set()
        try:
            self._q.put_nowait((None, None))
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        if self._reader is not None:
            try:
                self._reader["proc"].kill()
            except Exception:  # noqa: BLE001
                pass
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        if self.show:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass


class StatusOverlay:
    """把 ASR 字幕 / TTS 播报 / 链路状态画到视频帧上（复刻网页右侧面板）。

    与网页的对应关系（`static/orchestrator-test.html`）：

      · **ASR 字幕** = **替换**语义（流式刷新同一行）—— 网页里是
        ``$('asrline').innerHTML = ...``，这里就是"每来一条 partial 就覆盖"。
      · **TTS 播报** = **追加**语义（多轮累积）—— 网页里是往 ``ttslog``
        ``appendChild``，这里保留最近 N 条滚动显示。
      · **状态栏** = 网页的「音频」「延迟」两个面板。

    ⚠️ 两者语义**不能混**：早先网页里共用一个元素，ASR 的替换会把播报
    记录冲掉。这里分开维护是同一个道理。

    ⚠️ 只在**开窗时**画。录出来的 mp4 里烧字幕是另一件事（用户没要），
    而且字幕内容是**服务端实时回的**，与视频时间轴不是严格对齐的。
    """

    # 半透明黑底的等价系数（0.65 不透明黑 → 乘 0.35）
    _DIM = 0.35

    def __init__(self, overlay: "FaceOverlay", max_tts_lines: int = 3,
                 scale: float = 1.0) -> None:
        self.ovl = overlay
        self.max_tts_lines = max(1, max_tts_lines)
        self.scale = scale
        # ASR 是替换语义 —— 只留当前这一条
        self.asr_text = ""
        self.asr_partial = False
        # ASR 的五个状态量快照（说/抢/信/完），见 asr/client.py 的 AsrState
        self.asr_state: dict = {}
        # TTS 是追加语义 —— 保留最近的几条
        self.tts_lines: List[str] = []
        # 状态栏
        self.aec_mode = "?"
        self.anchor = "none"
        self.delay_ms = 0.0
        self.erle = None
        self.ref_ratio = 0.0
        self.tts_playing = 0          # 已开始、还没收到 tts.end 的段数
        #: TTS **还要播多久**（秒），由主循环每帧写入。
        #: ⚠️ 不能用 `tts_playing` 判断"在不在播"：`tts.end` 只表示音频
        #: **送**完了，而流式下服务端是一次性把几十秒推完的 —— 实测
        #: tts.start 在 8.8s、tts.end 在 11.0s，但音频还要响三十多秒。
        #: 所以状态栏必须看**播放进度**，否则开播两秒后就显示"空闲"。
        self.tts_remaining_s = 0.0
        # 本轮流式播报的文本累计（`tts.delta` 逐段来，收尾时清空）
        self.tts_text: List[str] = []
        self.tts_played_s = 0.0

    # ------------------------------------------------------------------ #

    def set_asr(self, text: str, partial: bool, state: dict = None) -> None:
        """替换语义：流式刷新同一行。``state`` 是服务端归纳的五个状态量。"""
        self.asr_text = text or ""
        self.asr_partial = partial
        self.asr_state = state or {}

    def add_tts(self, text: str) -> None:
        """追加语义：新播报加一行（与网页 `ttslog` 一致）。"""
        if text:
            self.tts_lines.append(text)

    def append_tts(self, text: str) -> None:
        """把流式增量接到**当前这一行**末尾（与网页的 tts.delta 一致）。

        与 `add_tts` 的区别：那个新起一行，这个续写本行 —— 流式下一个回复
        会被切成很多段，每段一行会碎得没法看。
        """
        if not text:
            return
        if self.tts_lines:
            self.tts_lines[-1] += text
        else:
            self.tts_lines.append(text)
            del self.tts_lines[:-self.max_tts_lines]

    def set_stats(self, m: dict) -> None:
        self.aec_mode = m.get("aec_mode", self.aec_mode)
        src = m.get("anchor_source")
        if src and src != "none":
            self.anchor = src
        self.delay_ms = m.get("delay_ms", self.delay_ms)
        self.erle = m.get("erle_db", self.erle)
        self.ref_ratio = m.get("ref_nonzero_ratio", self.ref_ratio)

    # ------------------------------------------------------------------ #

    def _bar(self, img, y0: int, h: int, alpha: float = 0.55) -> int:
        """在 y0 处铺一条半透明黑底，返回它的底边。"""
        H, W = img.shape[:2]
        y1 = min(H, y0 + h)
        if y1 <= y0:
            return y0
        roi = img[y0:y1]
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return y1

    def draw(self, img) -> None:
        """就地把字幕 + 状态栏画到帧上。"""
        H, W = img.shape[:2]
        k = max(0.5, W / 1280.0) * self.scale     # 按帧宽自适应字号
        lh = int(round(26 * k))
        pad = int(round(10 * k))

        # ---- 底部：ASR 字幕（替换语义，最多两行）----
        asr = self.asr_text.strip()
        if asr:
            lines = self._wrap(asr, 42 if W < 1000 else 62)[-2:]
            y0 = H - pad - lh * (len(lines) + 1) - pad
            self._bar(img, y0, lh * len(lines) + pad * 2)
            for i, ln in enumerate(lines):
                col = (120, 200, 255) if self.asr_partial else (150, 255, 170)
                self.ovl._text(img, ln, (pad, y0 + pad + lh * (i + 1) - 4), col)

        # ---- 右上角：状态栏 ----
        # ⚠️ 锚点 / ERLE / ferend 这三项（回声对齐诊断）已按需求**注释掉** ——
        #    要复看时把下面三行取消注释、并恢复 `st` 里的对应项即可。
        st = [f"AEC {self.aec_mode}",
              f"D {self.delay_ms:.0f}ms",
              # f"锚点 {self.anchor}",
              ]
        # if self.erle is not None:
        #     st.append(f"ERLE {self.erle:.1f}dB")
        # st.append(f"ferend {self.ref_ratio*100:.0f}%")
        # ⚠️ 看**播放进度**不看到没到 tts.end —— 后者只表示音频送完了
        #    （见 tts_remaining_s 的说明），用它会在开播两秒后就显示"空闲"
        if self.tts_remaining_s > 0.05:
            st.append(f"TTS 播报中 {self.tts_remaining_s:.0f}s")
        elif self.tts_playing:
            st.append("TTS 连接中")       # 已 start、音频还没到
        else:
            st.append("TTS 空闲")
        # ASR 五个状态量里的四个档位（transcript 就是下面那行字幕）
        s = self.asr_state
        if s:
            st.append("说{} 抢{} 信{} 完{}".format(
                s.get("user_speaking_confidence", "-"),
                s.get("barge_in_confidence", "-"),
                s.get("asr_confidence", "-"),
                s.get("turn_complete_confidence", "-"),
            ))
        bw = int(round(210 * k))
        self._bar(img, pad, lh * len(st) + pad, 0.5)
        for i, ln in enumerate(st):
            col = (140, 255, 180)
            # 锚点不是 ack 时变色告警 —— 已随锚点项一起注释掉
            # if ln.startswith("锚点") and self.anchor != "ack":
            #     col = (120, 120, 255)
            self.ovl._text(img, ln, (W - bw + pad, pad + lh * i + lh - 6), col)

        # ---- 左上角：TTS 播报记录（追加语义）----
        if self.tts_lines:
            self._bar(img, pad, lh * len(self.tts_lines) + pad, 0.45)
            for i, ln in enumerate(self.tts_lines):
                self.ovl._text(img, "播 " + ln,
                               (pad, pad + lh * i + lh - 6), (170, 235, 255))

    @staticmethod
    def _wrap(text: str, n: int) -> List[str]:
        """按字数硬折行 —— 中文没有空格，按宽度量的收益有限。"""
        return [text[i:i + n] for i in range(0, len(text), n)] or [""]


@dataclass
class Response:
    """一次 TTS 播报。"""
    rid: str
    text: str = ""
    chunks: List[np.ndarray] = field(default_factory=list)   # 24k int16
    armed_ctx: float = 0.0
    first_at: float = 0.0
    ended: bool = False
    cancelled: bool = False
    played_samples: int = 0

    @property
    def samples(self) -> int:
        return sum(len(c) for c in self.chunks)

    @property
    def seconds(self) -> float:
        return self.samples / TTS_SR


class VirtualPlayer:
    """复刻 `static/orchestrator-test.html` 里 ``PcmPlayer`` 的调度语义。

    离线下没有真实声卡，但服务端的 armed 协议要的就是"**你打算什么时候
    开始播**"和"**实际播了多少**"这两个数 —— 只要有调度表就能算出来。

    与真机逐条对齐的语义：

      · ``begin_response`` 只初始化状态，**不承诺**（承诺见 ``arm_now``）
      · ``arm_now`` 在**第一块音频到达时**才算承诺时刻，因此必然在未来
      · 首块用承诺时刻；**后续块按 ``nextAt`` 顺排**
        （⚠️ 后续块也不能取承诺值 —— 那样所有块会叠在同一时刻同时播，
         听感是"一闪而过"）
      · ``played_samples`` 用**音频时钟**算实际播出的比例，不是猜
      · ``stop`` 掐断已排程的源并返回**本句**已播出的采样数
        （服务端 resize 就是按这个基准理解的）
    """

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.cur: Optional[Response] = None
        self.responses: List[Response] = []
        self._sources: List[dict] = []      # {at, dur, played_full}
        self._armed = 0.0
        self._first_at = 0.0
        self._lead_ms = 200
        self._cancelled = False             # 上一句是否被打断（决定下一句重排）

    # ------------------------------ 生命周期 ------------------------------ #

    def begin_response(self, rid: str, text: str, lead_ms: int) -> Response:
        """收到 ``tts.start``：只初始化，**不承诺起播时刻**。"""
        self.cur = Response(rid=rid, text=text)
        self.responses.append(self.cur)
        self._lead_ms = lead_ms or 200
        self._armed = 0.0
        self._first_at = 0.0
        if self._cancelled:
            # 打断后的新一句：排程指针从**当前时刻**重新开始，不能沿用
            # 停在旧句末尾的位置（真机踩过：新句被排到旧句之后）
            self._sources = []
            self._cancelled = False
        return self.cur

    def arm_now(self) -> float:
        """第一块音频到达时**承诺**起播时刻（ctx 秒）。"""
        if self._armed:
            return self._armed
        self._armed = self.clock.now() + self._lead_ms / 1000.0
        return self._armed

    def schedule(self, pcm24: np.ndarray) -> float:
        """排程一块 24k int16 PCM，返回它的起播时刻（ctx 秒）。"""
        dur = len(pcm24) / TTS_SR
        if not self._first_at:
            at = self.arm_now()
            self._first_at = at
            if self.cur is not None:
                self.cur.armed_ctx = at
                self.cur.first_at = at
        else:
            prev = self._sources[-1]
            at = prev["at"] + prev["dur"]       # nextAt 顺排
        self._sources.append({"at": at, "dur": dur, "played_full": False})
        return at

    def played_samples(self) -> int:
        """本句**实际播出了多少采样**（16k 基准）。

        用音频时钟算，不是猜 —— 服务端 `RefTrack.resize` 要的就是这个值。
        """
        now = self.clock.now()
        played = 0.0
        for s in self._sources:
            if s["played_full"]:
                played += s["dur"]
            elif now > s["at"]:
                played += min(s["dur"], now - s["at"])
            # now <= at：还没起播 → 不计
        return int(round(played * SR))

    def stop(self) -> int:
        """打断：掐断已排程的源，返回**本句**已播出的采样数。

        与真机 ``PcmPlayer.stop()`` 一致：未起播的整个丢弃，正在播的只保留
        已播出的部分，然后清空调度表（下一句从零重排）。
        """
        played = self.played_samples()
        self._sources = []
        self._cancelled = True
        return played

    def end_response(self) -> None:
        if self.cur is not None:
            self.cur.ended = True

    def all_ended(self) -> bool:
        """所有已开始的 response 都收到 ``tts.end`` 了吗。

        ⚠️ 收尾判断**不能**只看 `remaining_s()`：`tts.start` 到达时
        response 已入列、但音频还没来（`_sources` 为空），此时剩余是 0 ——
        只看剩余会在**第一块音频到达前**就判定"播完了"（实测 TTS 0 段）。
        必须同时要求"音频已经送完"（tts.end）。
        """
        return bool(self.responses) and all(r.ended or r.cancelled
                                            for r in self.responses)

    def remaining_s(self) -> float:
        """**还要播多久**（秒）。0 = 已经在播完了。

        ⚠️ `tts.end` 只表示"音频**送**完了"，**不代表播完了** —— 整段是
        排程播放的，送到时可能才刚起播。收尾时若只看 tts.end 就关连接，
        会**把还在播的回复掐断**（真机实测：最后一句没放完就退出了）。
        这里按调度表算真实剩余时长 —— 它是"什么时候可以收工"的唯一依据。
        """
        now = self.clock.now()
        end = max((s["at"] + s["dur"] for s in self._sources), default=0.0)
        return max(0.0, end - now)


def _fmt_asr_state(state: dict) -> str:
    """把 ASR 消息里的 ``state`` 排成一行短标签（控制台/日志用）。

    四个档位量（``transcript`` 就是那行字幕本身，不重复显示）：

      说 = user_speaking_confidence   用户正在出声的把握
      抢 = barge_in_confidence        抢话轮的把握（按本轮已识别字数爬档）
      信 = asr_confidence             本轮转写置信度
      完 = turn_complete_confidence   本轮已说完的把握
    """
    if not state:
        return ""
    return "[说{} 抢{} 信{} 完{}]".format(
        state.get("user_speaking_confidence", "-"),
        state.get("barge_in_confidence", "-"),
        state.get("asr_confidence", "-"),
        state.get("turn_complete_confidence", "-"),
    )


# ============================================================================
# 编排服务客户端
# ============================================================================

class OrchestratorReplayClient:
    """`/v1/orchestrator` 的 Python 客户端（扮演浏览器）。"""

    def __init__(self, url: str, clock: VirtualClock,
                 aec_mode: str = "service",
                 save_tts_dir: str = "",
                 verbose: bool = False,
                 speaker: Optional[Speaker] = None,
                 face_log_every: int = 5) -> None:
        self.url = url
        self.clock = clock
        self.aec_mode = aec_mode
        self.save_tts_dir = save_tts_dir
        self.verbose = verbose
        self.speaker = speaker
        self.face_log_every = max(1, face_log_every)

        self.ws = None
        self.session_id: Optional[str] = None
        self.lead_ms = 200
        self.closed = False
        self.error: Optional[str] = None

        self.player = VirtualPlayer(clock)
        self.epoch = int(clock.now() * 1000) % 2147483647 or 1

        # 统计
        self.asr_partials = 0
        self.asr_finals: List[str] = []
        # 本轮 TTS 播报的文本累计（流式下由 `tts.delta` 逐段拼起来，
        # 收尾/打断时清空）。与 StatusOverlay.tts_text 是**两样东西** ——
        # 那个是画到视频帧上的多轮累积，这个是本轮原文。
        self.tts_text: List[str] = []
        self.face_states = 0
        self.face_frames_sent = 0
        self.omni_frames_sent = 0
        # 叠加层用：只留**最新一条** face.state，逐帧重画 —— 与浏览器 canvas
        # 在两条 face.state 之间保持上一次形状的行为一致。
        self.armed_sent: List[Tuple[str, float]] = []
        # 字幕/状态栏（开窗时由 main 注入；不注入就是 None，一切照常）
        self.status: Optional["StatusOverlay"] = None
        self.face_latest: Optional[dict] = None
        self.face_latest_at = 0.0
        self.face_src_wh: Optional[Tuple[int, int]] = None
        self.anchor_sources: List[str] = []
        self.errors: List[str] = []
        self._last_stats: Optional[dict] = None

    # ------------------------------ 连接 ------------------------------ #

    async def connect(self, identity: Optional[dict] = None) -> dict:
        import websockets
        self.ws = await websockets.connect(self.url, max_size=64 * 1024 * 1024)
        await self.ws.send(json.dumps({
            "type": "session.start",
            "identity": identity or {"page": "orchestrator-replay"},
            "aec_mode": self.aec_mode,
            # ⚠️ 能力位：没有它服务端就不等 armed 承诺，参考轨退回预测落位
            "caps": ["playback_anchor"],
            "seeded_delay_samples": 0,
            "system_prompt": "",
        }, ensure_ascii=False))
        # 等 session.ready
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=20))
            if msg.get("type") == "session.ready":
                self.session_id = msg.get("session_id")
                self.lead_ms = int(msg.get("lead_ms") or 200)
                if self.verbose:
                    print(f"  会话建立 id={self.session_id} "
                          f"sample_rate={msg.get('sample_rate')} lead_ms={self.lead_ms}")
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(f"session.start 被拒: {msg}")

    # ------------------------------ 上行 ------------------------------ #

    async def send_audio(self, x: np.ndarray) -> None:
        """送一个 100ms mic 块（float32 16k）。

        ``ctx_time`` 是本块**首采样**的 ctx 时刻；服务端据此拟合
        ctx↔会话采样 映射。**必须 > 0**（0 是"没上报"的哨兵）。
        """
        await self.ws.send(json.dumps({
            "type": "audio",
            "audio_base64": b64f32(x),
            "t_ms": 0,
            "ctx_time": self.clock.now(),
            "epoch": self.epoch,
        }, ensure_ascii=False))

    async def send_video_face(self, jpeg: Optional[bytes]) -> None:
        if not jpeg:
            return
        await self.ws.send(json.dumps({
            "type": "video_face", "frame_base64": b64bytes(jpeg), "t_ms": 0,
        }, ensure_ascii=False))
        self.face_frames_sent += 1

    async def send_video_omni(self, jpeg: Optional[bytes]) -> None:
        if not jpeg:
            return
        await self.ws.send(json.dumps({
            "type": "video_omni", "frame_base64": b64bytes(jpeg), "t_ms": 0,
        }, ensure_ascii=False))
        self.omni_frames_sent += 1

    async def send_playback(self, rid: str, phase: str,
                            sample_offset: int = 0,
                            start_ctx: float = 0.0,
                            stop_ctx: float = 0.0) -> None:
        await self.ws.send(json.dumps({
            "type": "playback", "response_id": rid, "phase": phase,
            "ctx_time": self.clock.now(), "seq": 0,
            "sample_offset": int(sample_offset),
            "start_ctx": float(start_ctx), "stop_ctx": float(stop_ctx),
            "epoch": self.epoch,
        }, ensure_ascii=False))

    async def request_stop(self) -> None:
        try:
            await self.ws.send(json.dumps({"type": "session.stop"}))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------ 下行 ------------------------------ #

    async def receive_loop(self) -> None:
        import websockets
        while not self.closed:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=90)
            except (asyncio.TimeoutError, websockets.ConnectionClosed, ConnectionError):
                break
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                break
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            # ⚠️ 单条消息处理失败**不能**让整个接收循环死掉。
            #    踩过：`_handle` 里访问了一个没定义的属性，抛 AttributeError
            #    直接把 receive_loop 干掉 —— 表现为**什么都收不到**（没 ASR、
            #    没 TTS），而错误只在 asyncio 的 "Task exception was never
            #    retrieved" 里一闪而过，极难定位。
            try:
                await self._handle(msg)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"处理 {msg.get('type')} 出错: "
                                   f"{type(exc).__name__}: {exc}")
                if self.verbose:
                    print(f"\n  ⚠️ 处理 {msg.get('type')} 失败: "
                          f"{type(exc).__name__}: {exc}")
        self.closed = True

    async def _handle(self, m: dict) -> None:
        t = m.get("type")
        if t == "tts.start":
            rid = m["response_id"]
            self.player.begin_response(rid, m.get("text", ""),
                                       int(m.get("lead_ms") or self.lead_ms))
            # ⚠️ 流式下 `tts.start` 的 text **必然是空的** —— 它是在 TTS
            # 合成之前发的。早先这里无条件 `add_tts(text)`，于是每轮都先压
            # 一行空的「播报」，真正的文字被挤到后面。现在只有非空才建行。
            if self.status is not None:
                if m.get("text"):
                    self.status.add_tts(m["text"])        # 追加语义（新起一行）
                self.status.tts_playing += 1
            if self.verbose:
                if m.get("text"):
                    print(f"\n  [tts.start {rid}] {m['text'][:60]}")
                else:
                    print(f"\n  [tts.start {rid}]（流式，文本随后到） ", end="", flush=True)

        elif t == "tts.audio":
            rid = m["response_id"]
            pcm = np.frombuffer(base64.b64decode(m["audio_base64"]), dtype=np.int16)
            first = not self.player._first_at
            at = self.player.schedule(pcm)
            if self.player.cur is not None:
                self.player.cur.chunks.append(pcm)
            # 真实出声（可选）。⚠️ 本工具不模拟声学回声，所以外放不会
            # 污染 mic 通道 —— 与真机不同，真机上喇叭一响麦克风就收得到。
            if self.speaker is not None:
                self.speaker.play(pcm)
            if first:
                # ⚠️ **承诺必须在这里发**（首块音频到达后），不能在 tts.start 时。
                #    服务端在 place() 参考轨之前等这个值（超时 250ms）。
                await self.send_playback(rid, "armed", start_ctx=at)
                # 自己记一笔：**我们发起承诺了**。服务端的 session.stats 是每
                # 2s 一次的快照，短会话里根本采不到播报那一刻的 ack 状态 ——
                # 只看快照会误判成"没走 ack"。这里是发起侧的权威记录。
                self.armed_sent.append((rid, at))
                # started 只用于服务端比对承诺是否兑现（告警，不修正）
                await self.send_playback(rid, "started", start_ctx=at)

        elif t == "tts.delta":
            # 流式文本增量。`tts.start` 是在 TTS 合成**之前**发的，那时
            # 还没有文本，所以整段合成能打印的那一行打印不出东西 ——
            # 文字只能从这里、或者收尾的 `tts.end` 拿。
            txt = m.get("text", "")
            if txt:
                self.tts_text.append(txt)
                if self.status is not None:
                    self.status.append_tts(txt)
                if self.verbose:
                    print(txt, end="", flush=True)

        elif t == "tts.end":
            rid = m["response_id"]
            # 整段合成（或流式收尾）在 `tts.end` 带全文；流式下文字已经由
            # `tts.delta` 逐段打印过，这里 text 可能为空。
            text = m.get("text") or ""
            if text and not self.tts_text:
                if self.status is not None:
                    self.status.add_tts(text)
                print(f"\n  [tts.end {rid}] {text[:80]}")
            elif self.tts_text:
                if self.verbose:
                    print(f"   [{rid} 播报完成 {len(''.join(self.tts_text))} 字]")
            self.tts_text = []
            self.player.end_response()
            if self.status is not None and self.status.tts_playing > 0:
                self.status.tts_playing -= 1
            await self.send_playback(rid, "ended")

        elif t == "tts.cancel":
            rid = m["response_id"]
            played = self.player.stop()
            if self.speaker is not None:
                self.speaker.stop()          # 丢弃还没播出去的缓冲
            if self.status is not None and self.status.tts_playing > 0:
                self.status.tts_playing -= 1
            if self.player.cur is not None:
                self.player.cur.cancelled = True
                self.player.cur.played_samples = played
            # 带上**实测播出量**与**实际停下时刻** —— 服务端据此把参考轨
            # 截到"真正播出过"的位置（resize 只能缩短，所以这个值要给准）
            await self.send_playback(rid, "cancelled", sample_offset=played,
                                     start_ctx=self.player._first_at,
                                     stop_ctx=self.clock.now())
            if self.verbose:
                print(f"\n  [tts.cancel {rid}] 已播 {played/SR:.2f}s")

        elif t == "asr":
            txt = m.get("text", "")
            state = m.get("state") or {}
            tag = _fmt_asr_state(state)
            if m.get("phase") == "partial":
                self.asr_partials += 1
                # 替换语义：流式刷新同一行（与网页 `$('asrline')` 一致）
                if self.status is not None:
                    self.status.set_asr(txt, True, state)
                if self.verbose:
                    print(f"\r  [ASR] {txt[:60]} {tag}", end="", flush=True)
            else:
                self.asr_finals.append(txt)
                if self.status is not None:
                    self.status.set_asr(txt, False, state)
                print(f"\n  [ASR final] {txt} {tag}")

        elif t == "session.stats":
            self._last_stats = m
            if self.status is not None:
                self.status.set_stats(m)
            src = m.get("anchor_source")
            # ⚠️ 回声对齐诊断（锚点状态记录 + 异常告警 + [stats] 那行）已按
            #    需求**注释掉** —— 这行 `[stats]` 每 2s 刷一次，会把 ASR 的
            #    流式输出冲掉。要复看时取消注释即可（`_last_stats` /
            #    `anchor_sources` 仍然在记，结尾汇总要用）。
            #
            # 只记**播报已经发生过**之后的锚点状态。
            #    `session.stats` 是每 2s 推一次的快照，会话刚开始、还没播报时
            #    它必然报 `none` —— 那是"还没轮到"，不是"用的是预测"。
            #    真机踩过这个歧义：只看最后一次 stats 会误判成没走 ack。
            if src and src != "none" and self.player.responses:
                if not self.anchor_sources or self.anchor_sources[-1] != src:
                    self.anchor_sources.append(src)
                # 这条是 AEC 对齐的命门：predicted = 服务端在**猜**播出时刻，
                # 误差逐句变化、固定 D 吸收不了。看到它就要查为什么没 ack。
                # if src != "ack":
                #     print(f"\n  ⚠️ 落位锚点 = {src}"
                #           f"（不是 ack —— 参考轨用于对齐的播出时刻不可信）")
            # if self.verbose:
            #     erle = m.get("erle_db")
            #     print(f"\r  [stats] D={m.get('delay_ms')}ms "
            #           f"锚点={src}(偏差{m.get('anchor_delta_ms')}ms) "
            #           f"ERLE={erle}dB ref非零={m.get('ref_nonzero_ratio')}", end="")

        elif t == "face.state":
            self.face_states += 1
            # 存**引用**不拷贝 —— 解析出来的 dict 之后没人改它
            self.face_latest = m
            self.face_latest_at = time.monotonic()
            tr = (m.get("tracks") or [None])[0]
            if tr and tr.get("src_w"):
                self.face_src_wh = (int(tr["src_w"]), int(tr["src_h"]))
            # 每 N 条打印一次人脸详情（默认 5 条，`--face-log-every` 可调）。
            # 节流的原因：服务端 face.state 是 ~10Hz，24fps 的输入下每帧都打
            # 会淹没其他日志。
            if self.verbose and self.face_states % self.face_log_every == 0:
                print("\n  " + self._fmt_face(m, self.face_states))

        elif t == "error":
            self.errors.append(f"{m.get('code')}: {m.get('message')}")
            print(f"\n  [ERROR] {m.get('code')}: {m.get('message')}")

    def playback_remaining_s(self) -> float:
        """TTS 还要播多久（秒）。收尾用它判断能否关连接。"""
        return self.player.remaining_s()

    def latest_face(self, hold_s: float = 0.4,
                    speed: float = 1.0) -> Optional[dict]:
        """取最近一条 face.state 供叠加层绘制；没有或已过期返回 None。

        ⚠️ **超时丢弃是必需的**：`face.state` 只在人脸线程产出观测时才推，
        流一停（会话结束 / 人脸线程异常 / 队列满被丢）最后那个框会**永久挂在
        画面上**。浏览器里看不出来（人眼不看长视频），录成 mp4 一眼就穿帮。

        ⚠️ 时限按**虚拟回放时间**折算：--replay-speed 2 时墙钟 0.2s 相当于
        视频里的 0.4s，不除 speed 就会在快放时把框留得过久。
        """
        if self.face_latest is None:
            return None
        if hold_s > 0 and (time.monotonic() - self.face_latest_at) > \
                hold_s / max(speed, 1e-6):
            return None
        # 人脸消失时服务端会推 valid=False —— 立刻停画，不等超时
        tr = (self.face_latest.get("tracks") or [None])[0]
        if not tr or not tr.get("valid"):
            return None
        return self.face_latest

    @staticmethod
    def _fmt_face(m: dict, n: int) -> str:
        """把一条 face.state 压成一行方便看。

        字段含义见 `orchestrator/protocol.py` 的 FaceDisplay：
        ``tracks[0]`` 是主说话人（box 是**服务端实际收到的帧尺寸**坐标系下的
        框，所以一并打出 src_w/src_h —— 曾因写死分辨率导致框错位）。
        """
        tr = (m.get("tracks") or [None])[0]
        if not tr or not tr.get("valid"):
            return f"[face #{n}] 未检测到人脸"
        box = tr.get("box")
        box_s = ("[%d,%d,%d,%d]" % tuple(int(v) for v in box)) if box else "-"
        parts = [
            f"[face #{n}]",
            f"置信={tr.get('score')}",
            f"框={box_s}",
            f"唇动={'说话中' if tr.get('speaking') else tr.get('lip')}",
            f"唤醒={'是' if tr.get('interacting') else '否'}",
        ]
        if tr.get("src_w"):
            parts.append(f"源={tr.get('src_w')}x{tr.get('src_h')}")
        ident = m.get("identity")
        if ident:
            parts.append(f"身份={ident.get('name') or '未注册'}"
                         f"(sim={ident.get('similarity')})")
        wake = m.get("wake")
        if wake:
            parts.append(f"最近唤醒={wake.get('phase')}/{wake.get('dwell_ms')}ms")
        fs = m.get("state")
        if fs:
            # G1 每帧 state（≈208ms 刷新一次）。几个档位与 ASR 侧同口径：
            # 脸 = face_present_confidence、唇 = lip_speaking_confidence、
            # 身份 = identity_confidence。
            parts.append(
                "state#%s[脸<%s> 唇<%s> 身份<%s> track=%s dwell=%sms 面积=%.1f%%%s]"
                % (fs.get("state_seq"),
                   fs.get("face_present_confidence"),
                   fs.get("lip_speaking_confidence"),
                   fs.get("identity_confidence"),
                   fs.get("track_id") if fs.get("track_id") is not None else "-",
                   fs.get("dwell_ms"),
                   float(fs.get("bbox_area_ratio") or 0.0) * 100,
                   (" 称呼=" + fs["display_name"]) if fs.get("display_name") else "")
            )
        return "  ".join(str(p) for p in parts)

    # ------------------------------ 收尾 ------------------------------ #

    async def close(self, save: bool = True) -> None:
        if save and self.save_tts_dir:
            self._save_tts()
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception:  # noqa: BLE001
            pass

    def _save_tts(self) -> None:
        os.makedirs(self.save_tts_dir, exist_ok=True)
        for i, r in enumerate(self.player.responses):
            if not r.chunks:
                continue
            x = np.concatenate(r.chunks)
            path = os.path.join(self.save_tts_dir, f"{i:02d}-{r.rid}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TTS_SR)
                w.writeframes(np.ascontiguousarray(x, dtype=np.int16).tobytes())
        print(f"  已保存 {sum(1 for r in self.player.responses if r.chunks)} "
              f"段 TTS 到 {self.save_tts_dir}/")


# ============================================================================
# 音视频提取
# ============================================================================

def resolve_asset(path: str) -> str:
    """把资源路径规整成**本机可读**的。

    动机：回放客户端既可能在服务端机器（106，仓库在 /data/...）跑，也可能
    在别的工作机上跑（仓库在别处），而 `--video assets/...` 这种相对路径
    两边都存在。直接 open 会因为 cwd 不同而失败，且报错信息很难看出是路径
    问题而不是文件问题。

    规则：能直接找到就用；否则按**脚本所在目录**再拼一次（这样从任意 cwd
    调用都行）。
    """
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(here, path)
    if os.path.exists(alt):
        return alt
    raise SystemExit("找不到资源：%s\n  （也试过 %s）" % (path, alt))


def load_audio(path: str, max_s: Optional[float] = None) -> np.ndarray:
    """读 wav/flac（soundfile）或从视频提音轨（ffmpeg）→ 16k float32。"""
    # 先按音频文件试（wav/flac/mp3 等）
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)                      # 多声道混合
        if sr != SR:
            x = _resample(x, sr, SR)
        return x[:int(max_s * SR)] if max_s else x.astype(np.float32)
    except Exception:  # noqa: BLE001 —— 不是音频文件，当视频处理
        pass
    cmd = ["ffmpeg", "-y", "-i", path, "-vn",
           "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1"]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += ["-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy()
    except subprocess.CalledProcessError:
        raise SystemExit(f"无法从 {path} 提取音频（ffmpeg 失败，且不是音频文件）")


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    n_out = int(len(x) * dst / src)
    pos = np.linspace(0, len(x) - 1, n_out)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    fr = (pos - i0).astype(np.float32)
    return (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)


def extract_frames(path: str, fps: float, max_w: int, max_h: int,
                   quality: int = 5) -> List[bytes]:
    """按 fps 抽 JPEG 帧，缩放到 max_w×max_h 以内。

    ⚠️ 缩放是**必须**的：浏览器发的就是缩放后的图（人脸 320×240、
    Omni 1280×720）。原图直接发会撑爆带宽与下游 KV。
    """
    vf = (f"fps={fps},scale={max_w}:{max_h}:force_original_aspect_ratio=decrease"
          f":force_divisible_by=2")
    cmd = ["ffmpeg", "-y", "-i", path, "-vf", vf, "-q:v", str(quality),
           "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return []
    frames, start = [], -1
    for i in range(len(out) - 1):
        if out[i] == 0xFF and out[i + 1] == 0xD8:
            start = i
        elif start >= 0 and out[i] == 0xFF and out[i + 1] == 0xD9:
            frames.append(out[start:i + 2])
            start = -1
    return frames


# ============================================================================
# 回放主流程
# ============================================================================

async def run_replay(client: OrchestratorReplayClient, audio: np.ndarray,
                     face_frames: List[bytes], omni_frames: List[bytes],
                     args, window: Optional["FaceWindow"] = None,
                     src_speaker: Optional["Speaker"] = None,
                     play_tts: bool = True, play_src: bool = True,
                     ) -> OrchestratorReplayClient:
    """按**真实实时节奏**回放：音频 100ms/块，人脸 24fps，Omni 1fps。

    三条流各自的节奏不同，所以用统一的 tick 循环按各自的截止时间触发 ——
    把视频塞进音频循环里最多只能到 10fps（100ms 一块）。
    """
    n_chunks = int(np.ceil(len(audio) / MIC_CHUNK))
    tail_chunks = int(args.tail_silence_s * 10)
    total_chunks = n_chunks + tail_chunks
    barge_at = {float(s) for s in args.bargein_at.split(",") if s.strip()}
    speed = max(args.replay_speed, 0.01)

    face_dt = 1.0 / max(args.face_fps, 0.1) if face_frames else 0.0
    omni_dt = 1.0 / max(args.omni_fps, 0.1) if omni_frames else 0.0

    print(f"\n=== 回放 {os.path.basename(args.audio or args.video)} ===")
    print(f"  音频 {len(audio)/SR:.1f}s（{n_chunks} 块 @100ms）+ 尾静音 "
          f"{args.tail_silence_s:.1f}s")
    if face_frames:
        print(f"  视频帧：人脸 {len(face_frames)} 帧 @{args.face_fps:.0f}fps"
              f"（{FACE_MAX_W}×{FACE_MAX_H}）、Omni {len(omni_frames)} 帧 "
              f"@{args.omni_fps:.0f}fps（{OMNI_MAX_W}×{OMNI_MAX_H}）")
    else:
        print("  视频：无（纯音频）")
    # 两个开关分别报，别再含糊地说"本地播放"
    _tts_ok = client.speaker is not None and client.speaker.ok
    _src_ok = ((window is not None and window._speaker is not None)
               or (src_speaker is not None and src_speaker.ok))
    print(f"  输入音频  ：{'开' if _src_ok else '关'}"
          + ("" if play_src else "（--no-play-audio）"))
    print(f"  TTS 声音  ：{'开' if _tts_ok else '关'}"
          + ("" if play_tts else "（--no-play-tts）"))
    if play_tts and not _tts_ok:
        err = client.speaker.err if client.speaker else "未启用"
        print(f"    ⚠️ TTS 放不出来：{err}")
        print("       pip install sounddevice（Linux 还需 libportaudio2）")
    print(f"  AEC 模式 = {args.aec_mode}　"
          f"{'（含 ' + str(len(barge_at)) + ' 次插话）' if barge_at else ''}")

    recv_task = asyncio.create_task(client.receive_loop())
    t0 = time.monotonic()
    idx = fi = oi = 0
    barge_seen: set = set()
    # 纯音频的输入音频播放是否因 TTS 在播而暂停（串行不重叠，见循环内说明）
    src_speaker_paused = [False]

    try:
        while idx < total_chunks and not client.closed:
            elapsed = time.monotonic() - t0
            now_v = elapsed * speed          # 虚拟回放进度（秒）

            # ── 音频：每 100ms 一块 ──
            if idx / 10.0 <= now_v:
                if idx < n_chunks:
                    chunk = audio[idx * MIC_CHUNK:(idx + 1) * MIC_CHUNK]
                    if len(chunk) < MIC_CHUNK:
                        chunk = np.pad(chunk, (0, MIC_CHUNK - len(chunk)))
                else:
                    # 尾静音：让 ASR 的 VAD 闭合最后一段
                    chunk = np.zeros(MIC_CHUNK, dtype=np.float32)
                await client.send_audio(chunk.astype(np.float32))
                # ── 纯音频：把**同一块**输入音频也放出来 ──
                # ⚠️ 必须**跟着 100ms 的块节奏**（外层这个 `if`），不能放在
                #    每 tick 都执行的地方 —— 主循环 TICK_S 只有 4ms，那样等于
                #    按 12.5 倍速灌音频，队列瞬间填满 → `play()` 开始丢最旧的，
                #    听感就是**卡顿/断续**（实测 3.75s 丢 111 帧、队列顶到 64）。
                #    这里喂的正好是本块 100ms，与真实时间 1:1。
                if src_speaker is not None and not src_speaker_paused[0]:
                    src_speaker.play(
                        (chunk * 32767.0).astype(np.int16))
                idx += 1

            # ── 人脸：24fps（约 41.7ms 一帧）──
            while face_frames and fi * face_dt <= now_v:
                await client.send_video_face(face_frames[fi % len(face_frames)])
                if window is not None:
                    # 带上**发这一帧时**最新的人脸状态；交给后台线程去
                    # 解码/绘制/编码，主循环只做一次非阻塞入队
                    window.push(
                        client.latest_face(args.face_hold_ms / 1000.0, speed),
                        client.face_src_wh)
                fi += 1

            # ── Omni：1fps ──
            while omni_frames and oi * omni_dt <= now_v:
                await client.send_video_omni(omni_frames[oi % len(omni_frames)])
                oi += 1

            # TTS **还要播多久** —— 状态栏与"让路"判据都用它。
            # ⚠️ 判"在不在播"必须看**播放进度**（remaining_s），不能看
            #    `tts_playing`（到没到 tts.end）也不能看 `pending_s()`
            #    （队列深度）：流式下服务端一次性把几十秒音频推完，
            #    tts.end 在开播两秒就到了，队列也会立刻排空 —— 两个都
            #    会误判成"空闲"，于是状态栏一直"空闲"、输入音频也不再让路。
            tts_left = client.player.remaining_s()
            if window is not None and window.status is not None:
                window.status.tts_remaining_s = tts_left

            # ── 纯音频：TTS 在播时让路（**串行不重叠**）──
            # 两个 Speaker 同时开会抢输出设备（实测互相打架、谁都出不来声）。
            # 注意这里**只切暂停状态**，真正喂音频在上面那个 100ms 块里
            # （每 tick 喂会变成 12.5 倍速 → 卡顿）。
            if src_speaker is not None:
                if tts_left > 0.05 and not src_speaker_paused[0]:
                    src_speaker.stop()          # 丢弃积压，TTS 优先
                    src_speaker_paused[0] = True
                elif tts_left <= 0.05 and src_speaker_paused[0]:
                    src_speaker_paused[0] = False

            # 插话：仅日志标记 —— 真正的打断由**服务端**新回复时的
            # `_interrupt_current` 驱动（它会发 tts.cancel），客户端据此停播。
            for b in barge_at:
                if b not in barge_seen and now_v >= b:
                    barge_seen.add(b)
                    print(f"\n  ── t={b:.1f}s 模拟插话（用户继续说话）──")

            # ⚠️ GUI 必须在**主线程**驱动：cv2 的 imshow/waitKey 从工作线程
            #    调会静默失效（窗口不刷新）。渲染线程只投帧，这里取最新一帧画。
            if window is not None and window.show:
                if not window.render():
                    window.show = False        # 用户按了 ESC

            slept = time.monotonic() - t0 - elapsed
            await asyncio.sleep(max(0.0, TICK_S - slept))
    except Exception:  # noqa: BLE001
        pass

    # ---- 收尾：等 TTS **真正播完**，不是等固定秒数 ----
    #
    # ⚠️ 这里踩过两个坑，本质是同一个：
    #   ① **不能一发 `session.stop` 就走** —— 服务端的 drain 会等 ASR/OmniLLM
    #      收尾，这一轮回复可能十几秒，TTS 音频还在陆续推过来。提前关连接
    #      会把后面的音频全丢掉，听感是"话说一半没了"。
    #   ② **不能只看 `tts.end`** —— 它只表示"音频**送**完了"，而整段是排程
    #      播放的，送到时可能才刚起播。要按**调度表**算"还要播多久"。
    # 所以：先等服务端把话说完（收到 tts.end 且输入已消费），再等播放器把
    # 缓冲放完，最后才关。
    print(f"\n  [收尾] 音频已发完，等回复生成 + 播放完毕"
          f"（最长 {args.drain_s:.0f}s）...")
    await client.request_stop()          # 让服务端开始 drain（它要等 ASR/OmniLLM）
    t_stop = time.monotonic()
    last_report = 0.0
    # ⚠️ 退出条件是「**收到过** TTS 且它已经播完」，不能只看剩余时长 ——
    #    服务端还在生成回复时 responses 本来就是空的、剩余也是 0，
    #    只看剩余会在**第一段 TTS 到达之前**就退出（实测 TTS 0 段）。
    #    另外还要留一个"一直没有 TTS"的兜底（比如这轮模型没回复）。
    got_any = False
    quiet_since = time.monotonic()
    # ⚠️ **不能是固定的墙钟上限**。早先写成 `while now - t_stop < drain_s`，
    #    60s 的长回复会在第 30s 被硬切 —— 与刚修好的"没播完就关"是同一类
    #    问题，只是触发条件是"回复比 drain_s 长"。
    #    正确语义：`drain_s` 是**没有进展时的静默容忍**，不是总时长上限。
    #    只要它还在播（或音频还在陆续到达），就一直等下去；真卡住了才退。
    #    另加一个很宽的总上限（drain_s × 6）兜住"服务端挂了但连接没断"。
    hard_limit = max(60.0, args.drain_s * 6.0)
    progress_at = time.monotonic()
    last_remain = None
    # ⚠️ **不能把 `client.closed` 当终止条件**。服务端 drain 完会**主动断开**
    #    （日志实测：client_stop 后 17s 关闭），但此时播放器队列里可能还排着
    #    十几秒没播的音频 —— 一断就退出会**把话掐断**，正是要修的症状。
    #    连接断了只是"没有新音频了"，本地把已收到的放完仍然要做。
    while True:
        now = time.monotonic()
        remain = client.playback_remaining_s()
        # 真出声时还要算上**队列里排着没播的**（Speaker 是异步消费的，
        # 调度表算不出来）
        if client.speaker is not None:
            remain = max(remain, client.speaker.pending_s())
        # ⚠️ **必须在这里也更新状态栏**。主循环那条在"音频发完"时就退出了，
        #    而 TTS 真正在播的几十秒**全在收尾这一段** —— 早先只更新主循环，
        #    于是收尾期间状态栏一直显示"空闲"（用户实测：渲染到视频上的
        #    TTS 状态一直是空闲）。
        if window is not None and window.status is not None:
            window.status.tts_remaining_s = remain
        if client.player.responses:
            got_any = True
            quiet_since = now
        # 收工条件：**音频已全部送达**（tts.end 齐）**且**播放器已放完。
        # 只看剩余时长会在首批音频到达前误判（见 all_ended 的说明）。
        if got_any and client.player.all_ended() and remain <= 0.05:
            break
        # "有进展" = 剩余时长变了，或音频条数变了 —— 都说明还在推进
        n_resp = len(client.player.responses)
        key = (round(remain, 1), n_resp)
        if key != last_remain:
            last_remain = key
            progress_at = now
        # 静默超时：drain_s 内毫无进展 → 认为结束了
        if now - progress_at > args.drain_s:
            print(f"    （{args.drain_s:.0f}s 无进展 —— 收工；"
                  f"收到 {n_resp} 段，剩余 {remain:.1f}s）")
            break
        if not got_any and now - quiet_since > 8.0:
            # 8s 内一段 TTS 都没来 —— 这轮大概没有回复，不必再等
            print("    （8s 内没有 TTS 到达 —— 本轮可能无回复，收工）")
            break
        if now - t_stop > hard_limit:
            print(f"  ⚠️ 收尾总时长超过 {hard_limit:.0f}s，强制退出")
            break
        if now - last_report > 2.0:
            last_report = now
            if remain > 0.05:
                print(f"    还在播：剩余 {remain:.1f}s"
                      f"（已收到 {n_resp} 段）", flush=True)
        # ⚠️ 收尾期间也要**持续投帧**，光调 render() 不够：
        #    `_run` 是从 `window.push()` 的队列里取活干的，队列空就不产新帧
        #    —— 而主循环在**视频发完**时就停了（`fi*face_dt <= now_v` 不再
        #    成立），偏偏 TTS 是在这之后才播的。于是状态栏停在最后一帧，
        #    「TTS 播放中」永远不出现。
        #    这里按 24fps 继续投，让字幕/状态栏跟着收尾阶段的进展走。
        if window is not None and (window.show or window.out_path):
            # 本循环 20Hz（sleep 0.05），每轮投一帧即 ~20fps —— 与主循环的
            # 24fps 同量级，够用；再高只会把只有 8 格的渲染队列塞满、白丢帧。
            window.push(client.latest_face(
                args.face_hold_ms / 1000.0, speed), client.face_src_wh)
        # ⚠️ 收尾期间也要驱动窗口 —— 否则等待的这几秒里窗口**无响应**
        #    （Windows 会画上"未响应"），而且最后一帧停在旧画面。
        if window is not None and window.show:
            window.render()
        await asyncio.sleep(0.05)

    remain = client.playback_remaining_s()
    if remain > 0.05:
        print(f"  ⚠️ 收尾超时，仍有 {remain:.1f}s 未播完（可调大 --drain-s）")

    if not recv_task.done():
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    return client


def _print_wrapped(text: str, indent: str = "    ",
                   hang: str = "") -> None:
    """按终端宽度折行打印全文（中文按 2 列宽算）。

    不截断 —— ASR/TTS 文本是核对链路正确性的主要依据，截断了就得去翻
    原始日志。
    """
    if not text:
        return
    try:
        import shutil
        cols = max(40, shutil.get_terminal_size((100, 24)).columns - 2)
    except Exception:  # noqa: BLE001
        cols = 98
    pad = len(indent)
    line, width = "", 0
    for ch in text:
        w = 2 if ord(ch) > 0x2E80 else 1        # 中日韩字符按 2 列
        if width + w > cols - pad:
            print(indent + line)
            indent, pad = (hang or indent), len(hang or indent)
            line, width = "", 0
        line += ch
        width += w
    if line:
        print(indent + line)


def print_summary(client: OrchestratorReplayClient, wall: float) -> None:
    print("\n" + "=" * 66)
    print("结果")
    print("-" * 66)
    rs = [r for r in client.player.responses if r.chunks]
    print(f"  TTS 回复      : {len(rs)} 段，共 {sum(r.seconds for r in rs):.1f}s")
    for i, r in enumerate(rs):
        mark = "（被打断）" if r.cancelled else ""
        print(f"    #{i+1} {r.rid}  {r.seconds:5.2f}s{mark}")
        # ⚠️ **不截断**：早先写死 text[:44] / t[:70]，长回复与长 ASR 只看到
        #    开头一截，没法核对内容。这里按终端宽度折行显示全文。
        _print_wrapped(r.text, indent="        ")
    print(f"  ASR           : {client.asr_partials} 个部分结果 / "
          f"{len(client.asr_finals)} 个最终结果")
    for t in client.asr_finals:
        _print_wrapped(t, indent="      · ", hang="        ")
    # ⚠️ 落位锚点汇总（AEC 对齐诊断）已按需求注释掉。`anchor_sources` 与
    #    `armed_sent` 仍在正常记录，要复看时取消注释即可。
    # if client.anchor_sources:
    #     # 这是 AEC 对齐的判据：ack = 用浏览器承诺的播出时刻（准）；
    #     # predicted = 服务端在猜（误差逐句变化，固定 D 吸收不了）
    #     print(f"  落位锚点      : {' → '.join(client.anchor_sources)}")
    #     if "predicted" in client.anchor_sources:
    #         print("    ⚠️ 出现过 predicted —— 参考轨落位不可信，AEC 对齐会漂")
    # elif rs and client.armed_sent:
    #     # 我们**自己发过** armed 承诺 —— 发起侧才是权威。
    #     # 服务端的 session.stats 是每 2s 一次的快照，短会话里采不到播报那一刻，
    #     # 只看它会误判成"没走 ack"。
    #     print(f"  落位锚点      : 已发出 {len(client.armed_sent)} 次 armed 承诺"
    #           f"（服务端 {client._last_stats.get('anchor_source', '?') if client._last_stats else '?'}）"
    #           f"—— 服务端日志里看『落位于 …（来源=ack）』确认")
    if client.face_states:
        print(f"  人脸状态      : {client.face_states} 条")
    if client._last_stats:
        s = client._last_stats
        # ERLE 属于回声对齐诊断，已随其余几项一起注释掉
        # print(f"  最终配置      : D={s.get('delay_ms')}ms"
        #       f"（来源 {s.get('delay_source')}）ERLE={s.get('erle_db')}dB")
        print(f"  最终配置      : D={s.get('delay_ms')}ms"
              f"（来源 {s.get('delay_source')}）")
    if client.errors:
        print(f"  错误          : {client.errors}")
    print(f"  墙钟          : {wall:.1f}s")
    print("=" * 66)


async def main_async(args) -> int:
    src = resolve_asset(args.audio or args.video)
    if args.video:
        args.video = resolve_asset(args.video)
    audio = load_audio(src, max_s=args.max_audio_s)
    if args.audio_seconds:
        audio = audio[:int(args.audio_seconds * SR)]

    face_frames: List[bytes] = []
    omni_frames: List[bytes] = []
    if args.video:
        print("  抽帧中（ffmpeg）...", flush=True)
        face_frames = extract_frames(args.video, args.face_fps, FACE_MAX_W, FACE_MAX_H, 5)
        omni_frames = extract_frames(args.video, args.omni_fps, OMNI_MAX_W, OMNI_MAX_H, 5)

    url = f"ws://{args.host}:{args.port}/v1/orchestrator"
    print(f"连接 {url}")
    clock = VirtualClock()

    # ---- 播放开关：默认全开，`--mute` 一次关掉 ----
    play_tts = args.play_all and args.play_tts
    play_src = args.play_all and args.play_audio

    speaker: Optional[Speaker] = None
    if play_tts:
        speaker = Speaker(device=args.audio_device)
        if not speaker.ok:
            print(f"  ⚠️ TTS 播放不可用：{speaker.err}")
            print("     装依赖：pip install sounddevice"
                  "（Linux 还需 libportaudio2）")
            speaker = None

    # 纯音频输入**没有窗口**，原片声没地方放（原来只有 FaceWindow 会播它）。
    # 这里补一个 16k 的 Speaker 专放输入音频。
    # ⚠️ **只在没有 --video 时创建** —— 有视频时仍走 FaceWindow 那条老路径
    #    （两个 Speaker 会抢输出设备，实测互相打架）。
    src_speaker: Optional[Speaker] = None
    if play_src and not args.video:
        src_speaker = Speaker(device=args.audio_device, sample_rate=SR)
        if not src_speaker.ok:
            print(f"  ⚠️ 输入音频播放不可用：{src_speaker.err}")
            src_speaker = None

    client = OrchestratorReplayClient(
        url, clock, aec_mode=args.aec_mode,
        save_tts_dir=args.save_tts, verbose=args.verbose,
        speaker=speaker, face_log_every=args.face_log_every)

    window: Optional[FaceWindow] = None
    # ⚠️ `--show` 现在默认开，所以**必须**先确认有显示器：cv2 在无头机上
    #    不是抛异常而是直接 abort，把整个进程带走（try/except 拦不住）。
    want_show = args.show
    if want_show and not _has_display():
        print("  ⚠️ 没检测到显示器 —— 自动不开窗（继续跑，只是看不到画面）")
        want_show = False
    if args.video and (want_show or args.save_video):
        ovl = FaceOverlay(font_path=args.overlay_font,
                          scale=args.overlay_scale,
                          force_ascii=args.overlay_ascii)
        if not ovl.ascii_only:
            print("  人脸标签：中文（Pillow + 中文字体）")
        else:
            print("  人脸标签：ASCII 回退（缺 Pillow 或中文字体；"
                  "pip install pillow 可显示中文）")
        # 字幕/状态栏（只在开窗时画 —— 见 StatusOverlay 的说明）
        status = StatusOverlay(ovl, scale=args.overlay_scale)
        client.status = status
        window = FaceWindow(args.video, args.save_video, args.face_fps,
                            show=want_show, overlay=ovl, crf=args.video_crf,
                            preset=args.video_preset,
                            play_audio=not args.no_source_audio,
                            mux_audio=not args.no_source_audio,
                            play_speed=args.replay_speed, status=status,
                            disp_scale=args.disp_scale)
        window.start()
        if not window.ok:
            print(f"  ⚠️ 回放窗口/录像不可用：{window.err}")
            window = None
        elif not window.out_path:
            # 说清楚：开了窗但**不会存文件**。早先不提示，用户跑完去找
            # 视频找不到（"帧"那个计数只表示渲染过，不代表落盘）。
            print("  （只显示、不存文件；要存成 mp4 加 --save-video 路径.mp4）")

    t0 = time.monotonic()
    try:
        await client.connect()
        await run_replay(client, audio, face_frames, omni_frames, args, window,
                         src_speaker=src_speaker,
                         play_tts=play_tts, play_src=play_src)
    except KeyboardInterrupt:
        print("\n  [中断]")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [失败] {type(exc).__name__}: {exc}")
        client.error = str(exc)
    finally:
        if window is not None:
            window.close()
            if window.frames:
                # ⚠️ 措辞要分清「渲染」与「落盘」：早先这里一律打"已写 N 帧"，
                #    而没传 --save-video 时根本**没写文件** —— 只显示在窗口上。
                #    用户据此去找文件，自然找不到。
                dst = (f" → {window.out_path}") if window.out_path else \
                    "（未落盘：没传 --save-video，只显示在窗口上）"
                print(f"  渲染 {window.frames} 帧" + dst
                      + (f"；丢弃 {window.dropped} 帧（渲染跟不上，"
                         f"只影响显示/录像，**不影响发给服务端的帧**）"
                         if window.dropped else ""))
        if speaker is not None:
            speaker.close()
        if src_speaker is not None:
            src_speaker.close()
        await client.close(save=True)

    print_summary(client, time.monotonic() - t0)
    return 1 if client.error else 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="编排服务离线音视频回放客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("## 用法")[-1] if __doc__ else "")
    # 位置参数：直接给文件，按扩展名自动判断音频/视频 —— 省掉 --audio/--video
    p.add_argument("source", nargs="?", default="",
                   help="输入文件（wav/flac/mp4...）。按扩展名自动判断"
                        "音频还是视频 —— 等价于 --audio/--video")
    p.add_argument("--audio", default="", help="音频文件（与 --video 二选一）")
    p.add_argument("--video", default="", help="视频文件（与 --audio 二选一）")
    p.add_argument("--host", default="127.0.0.1", help="编排服务地址（默认本机）")
    p.add_argument("--port", type=int, default=8100, help="编排服务端口（默认 8100）")
    p.add_argument("--aec-mode", default="service",
                   choices=["service", "browser", "off"],
                   help="回声消除模式（service=算法服务 AEC，默认）")
    p.add_argument("--max-audio-s", type=float, default=None,
                   help="限制音频时长（秒），默认整段")
    p.add_argument("--audio-seconds", type=float, default=0.0,
                   help="只回放前 N 秒（0=全部）")
    p.add_argument("--replay-speed", type=float, default=1.0,
                   help="回放倍速（1.0=真实时间。⚠️ 服务端按实时流设计，"
                        "加速可能让 AEC/ASR 表现失真）")
    p.add_argument("--tail-silence-s", type=float, default=2.0,
                   help="音频发完后补发的尾静音（秒），让 VAD 闭合最后一段")
    p.add_argument("--drain-s", type=float, default=300.0,
                   help="收尾时「多久没有进展」才放弃（秒，默认 300）。"
                        "⚠️ 它**不是总时长上限** —— 只要 TTS 还在播就继续等，"
                        "所以长回复不会被掐断（另有 drain_s×6 的硬上限兜底）")
    p.add_argument("--bargein-at", default="",
                   help="在这些秒数模拟插话（逗号分隔），如 5,12")
    p.add_argument("--face-fps", type=float, default=DEFAULT_FACE_FPS,
                   help=f"人脸帧抽帧率（默认 {DEFAULT_FACE_FPS:.0f}）")
    p.add_argument("--omni-fps", type=float, default=DEFAULT_OMNI_FPS,
                   help=f"Omni 帧抽帧率（默认 {DEFAULT_OMNI_FPS:.0f}）")
    p.add_argument("--save-tts", default="", help="把服务端回来的 TTS 存成 wav 的目录")
    # ⚠️ 命名容易混，记住这两条就够：
    #   `--play-tts`   = 放 **TTS（机器人回复）**
    #   `--play-audio` = 放 **输入音频**（你喂进去那段原片声）
    #   两者都默认开；静音用 `--mute`。
    p.add_argument("--play-tts", dest="play_tts", action="store_true",
                   default=True, help="播放 TTS 回复（默认开）")
    p.add_argument("--no-play-tts", dest="play_tts", action="store_false",
                   help="不放 TTS")
    p.add_argument("--play-audio", dest="play_audio", action="store_true",
                   default=True, help="播放输入音频（默认开）")
    p.add_argument("--no-play-audio", dest="play_audio",
                   action="store_false", help="不放输入音频")
    p.add_argument("--mute", dest="play_all", action="store_false",
                   default=True, help="静音（等价于 --no-play-tts --no-play-audio）")
    p.add_argument("--audio-device", default=None,
                   help="播放设备名/编号（sounddevice 的 device 参数）")
    p.add_argument("--face-log-every", type=int, default=5,
                   help="verbose 下每 N 条 face.state 打印一次详情（默认 5）")
    # ── 回放窗口 / 录像 ──
    p.add_argument("--no-source-audio", action="store_true",
                   help="回放窗口/录像**不要**原片的声音（默认要：开窗时"
                        "外放原片音轨，录像时把它 mux 进 mp4）")
    p.add_argument("--disp-scale", type=float, default=0.0,
                   help="窗口显示缩放（0=按屏幕自动适应，默认；"
                        "窗口内按 f 可在适应/1:1 间切换）。"
                        "只影响显示，录像仍是全分辨率")
    p.add_argument("--show", dest="show", action="store_true", default=True,
                   help="开窗口显示（默认开；纯音频时窗口里只有字幕。"
                        "需 opencv-python；无显示器时自动跳过）")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="不开窗")
    p.add_argument("--save-video", default="",
                   help="把带叠加层的视频存成 mp4（需 opencv-python + ffmpeg）")
    p.add_argument("--video-crf", type=int, default=20,
                   help="x264 质量（越小越清晰越大，默认 20）")
    p.add_argument("--video-preset", default="veryfast",
                   help="x264 preset（默认 veryfast）")
    p.add_argument("--face-hold-ms", type=int, default=400,
                   help="人脸状态过期时限(ms)：流一停/人脸消失后超过这个时间"
                        "就不再画框（默认 400，防止框永久挂在画面上）")
    p.add_argument("--overlay-scale", type=float, default=1.0,
                   help="叠加层字号缩放（默认 1.0）")
    p.add_argument("--overlay-ascii", action="store_true",
                   help="强制用 ASCII 标签（cv2.putText 画不了中文）")
    p.add_argument("--overlay-font", default="",
                   help="指定中文 TTF/TTC 字体路径（默认自动探测）")
    # 默认打开 —— 这个工具的价值在于看协议/时序，静默跑没意义。
    # 要安静输出用 --no-verbose。
    p.add_argument("--verbose", dest="verbose", action="store_true",
                   default=True, help="逐事件打印（默认开）")
    p.add_argument("--no-verbose", dest="verbose", action="store_false",
                   help="关掉逐事件打印")
    args = p.parse_args()

    # 位置参数按扩展名分流到 --audio / --video
    if args.source:
        if args.audio or args.video:
            p.error("给了位置参数就不要再给 --audio/--video")
        ext = os.path.splitext(args.source)[1].lower()
        if ext in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"):
            args.video = args.source
        else:
            args.audio = args.source
    if not args.audio and not args.video:
        p.error("需要一个输入文件（位置参数），或 --audio / --video 之一")
    if args.audio and args.video:
        p.error("--audio 与 --video 二选一（视频会自动抽音轨）")

    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
