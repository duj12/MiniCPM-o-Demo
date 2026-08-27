#!/usr/bin/env python3
"""流式视频对话 Demo — 支持音视频文件回放 / 实时采集。

复用 MiniCPM-o-Demo 的全双工流式协议（/v1/realtime?mode=video）：
  - 音视频持续逐帧发送（"边听边看"，累积进 KV cache）
  - 后端类型（qwen3omni / minicpm）由服务端 session.created 的 active_model
    自动判定，turn_decision 按后端推导：qwen3omni→VAD+TurnSense 判决，
    minicpm→模型自主 speak/listen。无需、也没有 --backend / --turn-decision 参数。

用法：
  # 默认走 gateway（生产路径，wss + mode=video；本机 :8006）
  python streaming_chat_demo.py --video xxx.mp4

  # 纯音频交互模式（只有 --audio，无视频）：语音对话
  python streaming_chat_demo.py --audio assets/audio/xxx.wav

  # 单路回放（视频理解，后端自动判定；音频块默认 1s，可 --audio-chunk-ms 100 调细粒度）
  python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
      --prompt "你是一个多模态助手，请简练回复用户的问题。" \
      --host "192.168.89.106"  --port 8006

  # 并发压力测试（10 路，用完整音轨驱动真实多轮；GPU 显存/利用率监控）
  python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
      --prompt "你是一个多模态助手，请简练回复用户的问题。" \
      --host "192.168.89.106"  --port 8006 \
      --concurrency 10 --gpu-ids 0,1

  # 实时采集（麦克风 + 摄像头），后端自动判定
  python streaming_chat_demo.py --realtime


常用参数：
  --audio-chunk-ms  音频发送块大小(ms)，默认 1000（VAD 需 ≥25ms；100 更实时但更频繁）
  --kv-budget       单分句视频帧预算(token)，默认 20000（≈每路 KV 25600 的 78%）
  --max-audio-s     限制回放/并发使用的音频时长(秒)，默认全部
  --tail-silence-s  音频放完后补发的静音时长(秒)，默认 2.0（VAD 靠尾静音闭合最后一段）
  --drain-idle-s    收尾空闲兜底(秒)，默认 5.0（正常靠 response_id 回执判定，不等超时）
  --concurrency N   并发路数，>0 进入并发测试
  --host / --port   生产服务地址（默认 gateway 8006；22500=直连后端需端口映射）
  --direct-backend  直连后端 WS（绕过 gateway），如 ws://127.0.0.1:22500/backend

输出指标（每轮，文件回放与实时采集口径一致，见 TurnTracker）：
  - TTFT      : 首字延迟（服务端 turn.turnsense=complete 判定句尾 → 第一个文本 token）
  - in_audio_s: 本轮输入音频长度（秒）
  - in_video_s: 本轮输入视频长度（秒）
  - reply_s   : 纯生成耗时（首字 → 本轮结束）
  - speed_cps : 生成速度（字符/秒）
"""

import argparse
import asyncio
import base64
import json
import os
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# 音频/视频处理（可选依赖）
try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import cv2
except ImportError:
    cv2 = None

SAMPLE_RATE = 16000
FRAME_MS = 100
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 1600
CHUNK_MS = 1000                                       # 默认音频块节奏（1s，并发/实时模式用）
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000        # 16000


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TurnMetrics:
    """单轮对话的指标。"""
    turn_idx: int = 0
    ttft_s: Optional[float] = None        # 首字延迟
    in_audio_s: float = 0.0               # 输入音频长度
    in_video_s: float = 0.0               # 输入视频长度
    reply_s: float = 0.0                  # 回复总耗时
    reply_chars: int = 0                  # 回复字符数
    speed_cps: float = 0.0                # 生成速度
    model_state: str = ""                 # listening / speaking
    reply_text: str = ""                  # 完整回复文本
    is_done: bool = False                 # 已收到 response.done（回复完整）

    @property
    def summary(self) -> str:
        ttft_str = f"{self.ttft_s:.2f}s" if self.ttft_s is not None else "N/A"
        # speed_cps=0 表示样本不足以测速（见 TurnTracker._finish_turn），
        # 打 N/A 而不是 0ch/s，避免和"真的很慢"混淆。
        spd_str = f"{self.speed_cps:.0f}ch/s" if self.speed_cps > 0 else "速度N/A"
        return (
            f"turn#{self.turn_idx}: TTFT={ttft_str} "
            f"in_audio={self.in_audio_s:.1f}s in_video={self.in_video_s:.1f}s "
            f"reply={self.reply_s:.1f}s ({self.reply_chars}ch, "
            f"{spd_str}) state={self.model_state}"
        )


# ============================================================================
# 音视频文件提取（离线回放）
# ============================================================================

def extract_audio_pcm(video_path: str, max_s: Optional[float] = None) -> np.ndarray:
    """用 ffmpeg 提取视频音轨为 16kHz float32 mono PCM。无音轨返回全零。

    max_s: 限制提取的音频时长（秒）。None=提取完整音轨。
    注意：完整音轨 token 数与时长成正比（MiniCPM 约 25 tok/s 音频），
    长视频音频可能超出 KV 预算，需配合 --kv-budget 或增大后端 -c。
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
    ]
    if max_s is not None:
        cmd += ["-t", str(max_s)]
    cmd += ["-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32)
    except subprocess.CalledProcessError:
        # 无音轨 → 返回对应时长静音
        dur = probe_duration(video_path)
        cap = min(dur, max_s) if max_s is not None else dur
        return np.zeros(int(cap * SAMPLE_RATE), dtype=np.float32)


def probe_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip() or 30.0)
    except Exception:
        return 30.0


def extract_keyframes(video_path: str, n_frames: int = 4) -> List[bytes]:
    """均匀提取 N 帧 JPEG（每帧 ≈ 540 token，控制 KV 用量）。"""
    dur = probe_duration(video_path)
    frames = []
    for i in range(n_frames):
        pos = dur * (i + 1) / (n_frames + 1)
        cmd = [
            "ffmpeg", "-y", "-ss", str(pos), "-i", video_path,
            "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, check=True)
            if r.stdout:
                frames.append(r.stdout)
        except subprocess.CalledProcessError:
            continue
    return frames


def extract_frames_evenly(video_path: str, fps: float = 1.0, max_frames: int = 0) -> List[bytes]:
    """按 fps 提取 JPEG 帧序列（流式节奏）。

    max_frames=0 表示不限制总数（依赖下游 KV 预算保护决定发多少帧）。
    返回全部分帧；发送方按 KV 预算动态决定实际使用量。
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "5",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        # 拆分 MJEPG 流（每个帧以 FFD8 开头，FFD9 结尾）
        frames, start = [], -1
        for i in range(len(out) - 1):
            if out[i] == 0xFF and out[i + 1] == 0xD8:
                start = i
            elif start >= 0 and out[i] == 0xFF and out[i + 1] == 0xD9:
                frames.append(out[start:i + 2])
                start = -1
                if max_frames and len(frames) >= max_frames:
                    break
        return frames
    except subprocess.CalledProcessError:
        return []


def b64(data) -> str:
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32).tobytes()
    return base64.b64encode(data).decode()


# ============================================================================
# WebSocket 客户端
# ============================================================================

class StreamingChatClient:
    """封装 /v1/realtime（或直连后端 /backend）的流式对话会话。"""

    def __init__(self, url: str, ssl_ctx=None, max_size: int = 128 * 1024 * 1024,
                 direct: bool = False, backend: Optional[str] = None,
                 echo: bool = True,
                 audio_sent_s: Optional[Callable[[], float]] = None,
                 video_sent_s: Optional[Callable[[], float]] = None,
                 sent_pushes: Optional[Callable[[], int]] = None):
        self.url = url
        self.ssl_ctx = ssl_ctx
        self.max_size = max_size
        self.direct = direct
        # 后端类型：minicpm（free-duplex，回复以 listen 结束，response.done 只是 chunk
        # 边界）或 qwen3omni（turn-based，response.done 即完整回复结束）。
        # None=未知：由 init() 从服务端 session.created 的 active_model 自动判定
        # （worker 转发事件时带上，见 worker._send_runtime_event）。
        self.backend = backend
        self.ws = None
        self.session_id = None

        # ── 同时收发的核心（原 TurnTracker）──────────────
        # 累计发送量回调：audio_sent_s/video_sent_s 返回调用方已发的音视频秒数
        # （指标按轮差值计）；sent_pushes 返回已发音频块数（收尾判据）。
        # 不提供时回退到内部计数器（send_input 每次自增）。
        self.echo = echo
        self._audio_sent_s = audio_sent_s or (lambda: self._sent_chunks * CHUNK_MS / 1000.0)
        self._video_sent_s = video_sent_s or (lambda: 0.0)
        self._sent_pushes = 0           # 内部计数：已发音频块
        self._sent_chunks = 0           # 内部计数：已发块（音视频同拍）

        self.metrics: List[TurnMetrics] = []
        self.in_reply = False           # 服务端正在生成本轮回复
        self.closed = False             # 会话结束（session.closed / 连接断开）
        self.last_event_at = time.monotonic()
        self.turn_no = 0
        self.max_seq = 0                # 已确认处理的最大 push 序号
        self.triggers = 0               # 服务端自注入的触发 push 数

        self._trigger_at: Optional[float] = None
        self._first_delta_at: Optional[float] = None
        self._text = ""
        self._n_deltas = 0              # 本轮收到的文本 delta 数（测速样本量）
        self._printing = False          # 已打印"回复流"表头
        self._mark_audio = 0.0          # 上轮开始时的累计发送量
        self._mark_video = 0.0
        self._cur_audio_s = 0.0
        self._cur_video_s = 0.0
        # 事件自带 server_send_ts（后端 epoch 秒）。折算到本地单调基准：
        #   monotonic ≈ server_epoch + offset
        # 并发下客户端 event loop 会被多路挤压，本地到达时刻失真；用服务端
        # 发出时刻算 TTFT / reply_s 才准确。单路下两者偏移恒定，结果不变。
        self._epoch_offset = time.monotonic() - time.time()

    async def connect(self):
        """建立 WS 并等待队列握手完成。

        - 经 gateway: 等 session.queue_done
        - 直连后端 /backend: 无 queue_done，connect 直接返回
        """
        import websockets
        self.ws = await websockets.connect(
            self.url, ssl=self.ssl_ctx, max_size=self.max_size,
        )
        if self.direct:
            return None  # 直连后端：init 直接响应，无 queue 握手
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15))
                t = msg.get("type")
                if t in ("session.queue_done", "queue_done"):
                    return msg  # 队列就绪，等 init
                if t == "session.error":
                    raise RuntimeError(f"queue error: {msg}")
        except asyncio.TimeoutError:
            raise RuntimeError("no queue_done within timeout")

    async def init(self, mode: str = "full_duplex", system_prompt: str = "",
                   turn_decision: Optional[str] = None, use_tts: bool = False,
                   config: Optional[dict] = None) -> dict:
        """发送 session.init。

        turn_decision 不传（None）：由服务端按 active_model 推导默认判决
        （qwen3omni→vad_turnsense，minicpm→model），worker 是权威。客户端在
        session.created 拿到 active_model 后再推导本地使用的判决（force_listen）。
        """
        payload = {
            "mode": mode,
            "use_tts": use_tts,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        cfg = dict(config or {})
        if turn_decision is not None:
            cfg["turn_decision"] = turn_decision
        cfg.setdefault("force_listen_count", 0)  # 供 MiniCPM 后端；VAD 判决必须为 0
        payload["config"] = cfg
        await self.ws.send(json.dumps({"type": "session.init", "payload": payload}))
        ev = await self._recv()
        while ev.get("type") not in ("session.created", "initialized"):
            if ev.get("type") == "session.error":
                raise RuntimeError(f"init error: {ev}")
            ev = await self._recv()
        self.session_id = ev.get("session_id")
        # 服务端在 session.created 里回传权威的 active_model（worker 转发时附加，
        # 见 worker._send_runtime_event）。以此判定后端类型。
        server_model = ev.get("active_model")
        if server_model in ("minicpm", "qwen3omni"):
            self.backend = server_model
        elif self.backend is None:
            raise RuntimeError(
                "无法判定后端类型：session.created 未带 active_model")
        return ev

    async def _recv(self, timeout: float = 300.0) -> dict:
        import websockets
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"recv timeout after {timeout}s")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def send_input(self, *, audio_b64: str = "", video_frames: Optional[List[str]] = None,
                         text: str = "", force_listen: bool = False,
                         max_new_tokens: Optional[int] = None) -> None:
        """发送一帧输入（音频 base64 + 视频帧）。force_listen=true 时只累积不触发回复。"""
        inp: dict = {}
        if audio_b64:
            inp["audio"] = audio_b64
        if video_frames:
            inp["video_frames"] = video_frames
        if text:
            inp["text"] = text
        if force_listen:
            inp["force_listen"] = True
        if max_new_tokens:
            inp["max_new_tokens"] = max_new_tokens
        await self.ws.send(json.dumps({"type": "input.append", "input": inp}, ensure_ascii=False))
        self._sent_chunks += 1   # 内部计数：供回退的 audio_sent_s / sent_pushes 用

    # ------------------------------------------------------------------
    # 显式收发：send_input 由调用方逐帧调用；receive_loop 由调用方作为独立
    # task await。websockets 同连接不能同时 send/recv，所以调用方用
    # asyncio.gather 并发两个 task（一个发一个收），由调用方掌握收的时机。
    # ------------------------------------------------------------------

    async def receive_loop(self) -> None:
        """显式接收循环：持续收事件并处理，直到会话结束。调用方作为 task await。"""
        import websockets
        while not self.closed:
            try:
                ev = await self._recv(timeout=3600.0)
            except (TimeoutError, websockets.ConnectionClosed, ConnectionError):
                break
            except Exception as exc:
                if self.echo:
                    print(f"\n  [recv error] {exc}", flush=True)
                break
            if not self.handle_event(ev):
                break
        self.closed = True

    # ------------------------------------------------------------------
    # 回复事件处理（原 TurnTracker，已并入 Client）：按轮统计指标
    # ------------------------------------------------------------------

    def _finish_empty_turn(self, idle: float, reply_stall_s: float = 20.0,
                           empty_reply_quiet_s: float = 2.5) -> bool:
        """识别"触发了但模型无话可说"的空回复轮并收尾，返回是否收尾了。

        这种轮次不会有 response.done：触发块本身回的是 listen，服务端据此复位，
        客户端却在等一个永远不来的终止信号（原来要白等满 reply_stall_s）。

        判据要同时满足，避免把正常回复误判成空：
          1. 本轮一个文本 delta 都没有
          2. 所有已发 push 都已确认（没有排队中的输入还会产出东西）
          3. 静默 empty_reply_quiet_s —— 真在生成时要么出文本，要么还有回执在流动
        """
        if not self.in_reply or self._first_delta_at is not None:
            return False
        if not self.all_input_consumed() or idle < empty_reply_quiet_s:
            return False
        if self.echo:
            print(f"  [空回复] 本轮模型未输出文本（push 已全部确认），收尾", flush=True)
        self._finish_turn()
        return True

    async def wait_while_replying(self, reply_stall_s: float = 20.0,
                                  empty_reply_quiet_s: float = 2.5) -> None:
        """阻塞到本轮回复结束（--wait-reply 用）。不死等：空回复/停滞都会收尾。"""
        while self.in_reply and not self.closed:
            idle = time.monotonic() - self.last_event_at
            if self._finish_empty_turn(idle, reply_stall_s, empty_reply_quiet_s):
                break
            if idle > reply_stall_s:
                if self.echo:
                    print(f"\n  [warn] 回复停滞 >{reply_stall_s:.0f}s，强制收尾", flush=True)
                self._finish_turn()
                break
            await asyncio.sleep(0.05)

    def all_input_consumed(self) -> bool:
        """服务端是否已把客户端发出的输入都处理完 —— 不必等空闲超时。

        每个被处理的 push 恰好回一个终结事件（listen 或 response.done），其
        response_id 尾号是服务端严格递增的处理序号。服务端自己也注入 push
        （每次 turn.turnsense=complete 一个触发解码块），所以序号上界是
        "已发块数 + 触发数"。

        但上界取不到：half_duplex 在 GENERATING 期间收到的音频块会被直接丢弃
        （只用于 barge-in 检测，不 push 进 KV），因此每轮触发瞬间在途的那一块
        没有回执。实测 20 块 + 2 触发 → 序号停在 20。故用下界判据。
        """
        return self.max_seq >= self._sent_chunks

    async def drain(self, idle_s: float = 10.0, grace_s: float = 1.5,
                    max_s: float = 300.0,
                    reply_stall_s: float = 20.0,
                    empty_reply_quiet_s: float = 2.5) -> None:
        """收尾：等服务端把已发送的输入全部消费完，然后立刻返回。

        正常路径不靠超时：all_input_consumed() 追平即说明没有待处理输入了，
        再静候 grace_s 即可返回——留这点余量是因为 TurnSense 的 incomplete
        watchdog（--ts-wait-ms，默认 900ms）可能在最后一块之后才补触发一次回复。

        idle_s 只是兜底：序号对不上时（如 --wait-reply=false 边回复边发送）
        才退回"连续无事件"判据。in_reply 期间由 reply_stall_s 兜底，生成本身
        多慢都不会被掐断。
        """
        t0 = time.monotonic()
        while not self.closed and (time.monotonic() - t0) < max_s:
            idle = time.monotonic() - max(t0, self.last_event_at)
            if self.in_reply:
                if not self._finish_empty_turn(idle, reply_stall_s, empty_reply_quiet_s) \
                        and idle > reply_stall_s:
                    if self.echo:
                        print(f"\n  [warn] 回复停滞 >{reply_stall_s:.0f}s，强制收尾",
                              flush=True)
                    self._finish_turn()
            elif self.all_input_consumed():
                if idle >= grace_s:
                    if self.echo:
                        print(f"  [收尾] 输入已全部消费（seq={self.max_seq}），耗时 "
                              f"{time.monotonic() - t0:.1f}s", flush=True)
                    return
            elif idle > idle_s:
                if self.echo:
                    print(f"  [收尾] 空闲 {idle_s:.0f}s 兜底退出"
                          f"（seq={self.max_seq}/{self._sent_chunks + self.triggers}）",
                          flush=True)
                return
            await asyncio.sleep(0.05)

    def _ev_mono(self, ev: dict) -> Optional[float]:
        """事件的服务端发出时刻，折算到本地单调基准。无 server_send_ts 返回 None。"""
        ts = ev.get("server_send_ts")
        if ts is None:
            return None
        try:
            return float(ts) + self._epoch_offset
        except (TypeError, ValueError):
            return None

    def _begin_turn(self) -> None:
        if self.in_reply:
            return
        self.in_reply = True
        self.turn_no += 1
        a, v = self._audio_sent_s(), self._video_sent_s()
        self._cur_audio_s = max(0.0, a - self._mark_audio)
        self._cur_video_s = max(0.0, v - self._mark_video)
        self._mark_audio, self._mark_video = a, v

    def _finish_turn(self, final_text: str = "", end_ts: Optional[float] = None) -> None:
        if not self.in_reply:
            return
        text = final_text or self._text
        m = TurnMetrics(turn_idx=self.turn_no,
                        in_audio_s=self._cur_audio_s,
                        in_video_s=self._cur_video_s)
        m.reply_text = text
        m.reply_chars = len(text)
        m.model_state = "speaking" if text else "listening"
        m.is_done = True
        if self._first_delta_at is not None:
            if self._trigger_at is not None:
                m.ttft_s = self._first_delta_at - self._trigger_at
            # end_ts 优先用服务端发出时刻（response.done 的 server_send_ts），
            # 避免并发下客户端 event loop 挤压导致 reply_s≈0。
            m.reply_s = (end_ts or time.monotonic()) - self._first_delta_at
            # 只有回复真的是"流出来"的才报速度：至少 2 个 delta、跨度够长。
            # 退化轮次（如被 barge-in 打断，几十字挤在一次到达）算出来的
            # 是几千 ch/s 的假值，宁可不报。
            if self._n_deltas >= 2 and m.reply_s >= 0.05:
                m.speed_cps = m.reply_chars / m.reply_s
        self.metrics.append(m)
        if self.echo:
            if self._printing:
                print()          # 收束"回复流"那一行
            print(f"  {m.summary}", flush=True)

        self.in_reply = False
        self._trigger_at = None
        self._first_delta_at = None
        self._text = ""
        self._n_deltas = 0
        self._printing = False

    def handle_event(self, ev: dict) -> bool:
        """处理单个事件。返回 False 表示会话结束、receiver 应退出。"""
        self.last_event_at = time.monotonic()
        t = ev.get("type")

        # response_id = "{session_id}-{N}"，N 是服务端 push 处理序号（严格递增）。
        # 用它判断输入是否全部消费完，见 all_input_consumed()。
        rid = ev.get("response_id")
        if rid:
            try:
                self.max_seq = max(self.max_seq, int(str(rid).rsplit("-", 1)[-1]))
            except ValueError:
                pass

        if t == "turn.turnsense":
            label = ev.get("label")
            if label == "complete":
                self.triggers += 1   # 服务端随后会注入一个触发解码 push
                # 服务端判定句尾并触发生成 —— TTFT 的计时起点。
                # 在这里（而非首个 delta）置 in_reply：让发送侧在 prefill 阶段
                # 就暂停，避免后续音频被 VAD 当成 barge-in 打断当前回复。
                # 用服务端时间戳（worker 直发时已带 server_send_ts）：与后端
                # delta 的 server_send_ts 同时钟，并发下 TTFT 才不会因 client
                # event-loop 挤压或队列延迟变负值。
                self._begin_turn()
                self._trigger_at = self._ev_mono(ev) or time.monotonic()
                if self.echo:
                    print(f"\n  [VAD段{self.turn_no}] 判定句尾 → 模型开始回复", flush=True)
            elif self.echo:
                # incomplete = 语义不完整，等用户续说；invalid = 噪音/过短，丢弃
                print(f"\n  [turnsense] {label}", flush=True)

        elif t == "response.output.delta":
            kind = ev.get("kind")
            if kind == "text":
                txt = ev.get("text", "")
                if txt:
                    self._begin_turn()   # model 判决无 turnsense 事件，在此开轮
                    if self._first_delta_at is None:
                        # 用服务端发出时刻（server_send_ts）算首字时间，
                        # 并发下避免 event loop 挤压导致 TTFT 失真。
                        self._first_delta_at = self._ev_mono(ev) or time.monotonic()
                    self._n_deltas += 1
                    self._text += txt
                    if self.echo:
                        if not self._printing:
                            self._printing = True
                            print("  ── 回复流 ── ", end="", flush=True)
                        print(txt, end="", flush=True)
            elif kind == "listen":
                # MiniCPM free-duplex: <|listen|> = 模型让出话轮，本轮结束。
                # Qwen3-Omni: listen 是每个 force_listen 块的 prefill 回执
                # （1 块 1 个，贯穿整个聆听期），不能据此收尾——触发瞬间还有
                # 在途块的回执会到，据此收尾会把回复截断在开头。
                # 触发块自己回 listen（模型无话可说、没有 response.done）的情况
                # 由 _finish_empty_turn 用"静默 + 全部确认"识别。
                if self.backend == "minicpm":
                    self._finish_turn()
            elif kind == "audio":
                pass

        elif t == "response.done":
            # Qwen3-Omni turn-based: done = 完整回复结束。
            # MiniCPM free-duplex: done 只是 chunk 边界（文本已由 delta 累积），
            # listen 才是 turn 结束——按 done 收尾会把一轮拆成多轮。
            if self.backend != "minicpm":
                self._begin_turn()
                self._finish_turn(ev.get("text", "") or "", end_ts=self._ev_mono(ev))

        elif t == "session.closed":
            self._finish_turn()
            return False

        elif t == "session.error":
            print(f"\n  [error] {ev}", flush=True)

        return True

    async def close(self, reason: str = "demo_done"):
        # 直连后端：用 HTTP POST /sessions/{id}/close 可靠释放 session
        # （后端 WS 不处理 session.close 消息，发 WS close 可能导致 session 泄漏）。
        if self.direct and self.session_id:
            try:
                import httpx
                base = self.url.split("/backend")[0]
                async with httpx.AsyncClient() as c:
                    await c.post(f"{base}/sessions/{self.session_id}/close",
                                 json={"reason": reason})
            except Exception:
                pass
        try:
            await self.ws.send(json.dumps({"type": "session.close", "reason": reason}))
        except Exception:
            pass
        try:
            await self.ws.close()
        except Exception:
            pass


# ============================================================================
# 文件回放模式
# ============================================================================

async def run_file_replay(client: StreamingChatClient, video_path: str,
                          prompt: str,
                          fps: float = 1.0, max_frames: int = 0,
                          kv_budget_tokens: int = 20000,
                          audio_path: str = "",
                          max_audio_s: Optional[float] = None,
                          replay_speed: float = 1.0,
                          wait_reply: bool = True,
                          audio_chunk_ms: int = 1000,
                          max_new_tokens: int = 1024,
                          tail_silence_s: float = 2.0,
                          drain_idle_s: float = 5.0):
    """离线：按 fps 流式抽帧 + 音频切块发送，模拟实时流。

    帧策略：按 fps 持续抽帧（max_frames=0 不限制总数），但用 KV 预算保护
    ——每帧 ≈ 540 token（缩放后），累计超过 kv_budget_tokens 后停止发帧
    （音频继续），避免撑爆 KV cache。这样"最新画面"始终在流，旧帧滚动丢弃。

    后端类型与 turn_decision 由 client.backend 推导（init 后已赋值）：
    qwen3omni→vad_turnsense，minicpm→model。
    audio_path: 指定人声音频 wav（替代视频音轨）。视频音轨常为非人声背景音，
      TurnSense 会判 invalid；用清晰人声可验证 VAD+TurnSense 完整链路。
    max_audio_s: 限制音频提取时长（秒）。None=完整音轨。注意长音频 token
      量巨大（约 25 tok/s），可能超出 KV，需配合 --kv-budget / 后端 -c。
    tail_silence_s: 音频放完后补发的静音时长。FSMN VAD 靠尾静音闭合语音段，
      音频一放完就停发，最后一句可能永远不触发回复。
    drain_idle_s: 收尾等待。发完后只收不发，连续这么久无事件才判定结束——
      否则最后一段回复还在生成中就被 close() 掐断（收不到结果）。
    """
    # 后端类型与 turn_decision 由 client.backend 推导（init 后已赋值）：
    # qwen3omni→vad_turnsense（VAD+TurnSense 分句触发），minicpm→model（模型自主）。
    if client.backend == "qwen3omni":
        turn_decision = "vad_turnsense"
    else:
        turn_decision = "model"

    # 回放模式音频块节奏：默认 1000ms（由 --audio-chunk-ms 控制，可调 100ms 更实时）。
    # 视频帧节奏与音频解耦：每 _frames_per_audio_chunk 个音频块（1s）发 1 帧，
    # 保持视频 1s 一帧不爆 KV。并发/实时模式仍用全局 CHUNK_MS。
    _chunk_ms = max(25, audio_chunk_ms)
    _chunk_samples = SAMPLE_RATE * _chunk_ms // 1000
    _frames_per_audio_chunk = max(1, 1000 // _chunk_ms)  # 默认 10 → 每 1s 发 1 帧

    # 音频来源：优先 --audio（人声 wav），否则从视频提取音轨。纯音频模式
    # （video_path 为空）必须有 --audio。
    if audio_path and os.path.exists(audio_path):
        if sf is None:
            print("  [warn] soundfile 不可用，忽略 --audio")
        else:
            a, sr = sf.read(audio_path, dtype="float32")
            if sr != SAMPLE_RATE:
                print(f"  [warn] --audio 采样率 {sr}≠16000，将按原采样率处理（可能不准）")
            audio = np.asarray(a, dtype=np.float32)
    else:
        audio = extract_audio_pcm(video_path, max_s=max_audio_s)

    # 纯音频模式：无视频帧
    frames = extract_frames_evenly(video_path, fps=fps, max_frames=max_frames) if video_path else []
    video_s = probe_duration(video_path) if video_path else 0.0
    audio_s = len(audio) / SAMPLE_RATE

    src_name = os.path.basename(audio_path or video_path) or "(audio)"
    print(f"\n=== 文件回放: {src_name} ===")
    print(f"  音频 {audio_s:.1f}s, 视频 {video_s:.1f}s, 抽帧 {len(frames)} 帧 @{fps}fps"
          + (" [纯音频模式]" if not video_path else ""))
    print(f"  turn_decision = {turn_decision}, KV预算 ≈{kv_budget_tokens} tok")

    TOKENS_PER_FRAME = 540  # 1440p 缩放后每帧约 540 token

    n_chunks = max(1, int(np.ceil(len(audio) / _chunk_samples)))
    # 尾静音块：音频放完后继续喂静音，让 VAD 能闭合最后一个语音段。
    n_tail = max(0, int(round(tail_silence_s * 1000 / _chunk_ms)))
    silence = np.zeros(_chunk_samples, dtype=np.float32)
    print(f"  尾静音 {tail_silence_s:.1f}s ({n_tail} 块), 收尾空闲判据 {drain_idle_s:.0f}s")

    sent_frames = 0     # 已发送视频帧数

    # ── 显式双 task：一个发送、一个接收，都由调用方创建，asyncio.gather 并发 ──
    # 发送 task 只管按节奏 send_input；接收 task 是 client.receive_loop（阻塞收，
    # 到达即处理）。两者交替跑在同一个 event loop 上。指标回调注入 client：
    # audio_sent_s 由 client 内部计数回退，video 用这里的帧计数。
    client._video_sent_s = lambda: sent_frames / max(fps, 0.1)

    async def send_loop():
        nonlocal sent_frames   # sent_frames 是闭包外部变量（跨轮累计），send_loop 内自增
        turn_frames_0 = 0      # 本轮分句起始的帧计数（KV 预算按分句算，不跨轮累计）
        last_turn_no = 0
        for idx in range(n_chunks + n_tail):
            if client.closed:
                break
            # VAD+TurnSense + wait_reply：回复期间暂停发送，避免新音频被 VAD
            # 当成 barge-in 打断当前回复。
            if wait_reply and turn_decision == "vad_turnsense":
                await client.wait_while_replying()
                if client.closed:
                    break
            # 新一轮开始 → 重置本分句的视频帧预算
            if client.turn_no != last_turn_no:
                last_turn_no = client.turn_no
                turn_frames_0 = sent_frames

            if idx < n_chunks:
                chunk = audio[idx * _chunk_samples:(idx + 1) * _chunk_samples]
                if len(chunk) < _chunk_samples:
                    chunk = np.pad(chunk, (0, _chunk_samples - len(chunk)))
            else:
                chunk = silence   # 尾静音，只为触发 VAD 尾判定

            frame_b64 = []
            # 视频帧节奏独立于音频：每 _frames_per_audio_chunk 个音频块（1s）发 1 帧。
            # 这样音频可 100ms 细粒度流式（VAD 更实时），而视频保持 1s 一帧不爆 KV。
            # kv_budget 限制的是"当前分句"发送的视频帧数（单次持续处理的帧预算），
            # 不是累计发送帧数——否则多轮对话后预算被跨分句耗尽，视频帧提前停发。
            if idx < n_chunks and frames and idx % _frames_per_audio_chunk == 0 and \
               (sent_frames - turn_frames_0) * TOKENS_PER_FRAME < kv_budget_tokens:
                frame_b64 = [b64(frames[sent_frames % len(frames)])]
                sent_frames += 1

            # force_listen 只用于 VAD+TurnSense 判决（worker 累积音频，VAD 触发才回复）。
            # 自由全双工（turn_decision=model）下必须让模型自主 listen/speak，
            # 不能 force_listen，否则模型永远只累积不回复。
            await client.send_input(
                audio_b64=b64(chunk), video_frames=frame_b64,
                text=prompt if idx == 0 else "",
                force_listen=(turn_decision == "vad_turnsense"),
            )
            await asyncio.sleep(_chunk_ms / 1000.0 / replay_speed)

    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(client.receive_loop())
    # 发送完/会话结束后：收尾等剩余回复（send 结束后 receive_loop 仍在收）。
    await send_task
    if not client.closed:
        print(f"\n  [收尾] 音频发送完毕，等待剩余回复"
              f"（{drain_idle_s:.0f}s 无事件即结束）...", flush=True)
        await client.drain(idle_s=drain_idle_s)
    # 会话已结束则接收循环也退出了；否则取消它（drain 已完成）。
    if not recv_task.done():
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass

    return client.metrics


# ============================================================================
# 实时采集模式
# ============================================================================

async def run_realtime(client: StreamingChatClient, system_prompt: str,
                       turn_decision: str, duration_s: float = 30.0,
                       fps: float = 1.0, kv_budget_tokens: int = 20000,
                       tail_silence_s: float = 2.0,
                       drain_idle_s: float = 5.0):
    """在线：麦克风 + 摄像头实时采集。需要 sounddevice + opencv。

    与文件回放共用 client 的 receiver task，指标口径一致。
    视频按 fps 抽帧（KV 预算保护：接近上限后只发音频，保留最新画面）。

    与回放的区别：麦克风在模型回复期间**不暂停**——真实双工场景下用户随时可
    插话（barge-in 由服务端 VAD 处理），暂停会让采集与时间轴脱节。
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("实时采集需要 sounddevice: pip install sounddevice")
        return []

    print(f"\n=== 实时采集（{duration_s}s）===")
    print(f"  turn_decision = {turn_decision}")
    print("  按 Enter 开始...")
    input()

    cap = cv2.VideoCapture(0) if cv2 else None
    if cap and not cap.isOpened():
        cap = None
    if cap is None:
        print("  [warn] 摄像头不可用，仅发送音频")

    TOKENS_PER_FRAME = 540
    sent_frames = 0     # 已发送视频帧数

    # 指标回调：audio 用 client 内部计数回退；video 用这里的帧计数。
    client._video_sent_s = lambda: sent_frames / max(fps, 0.1)

    # sounddevice 回调在音频线程里跑：只做 deque.append（GIL 下原子），
    # 不碰 numpy 拼接，避免与主协程的切片竞争。
    from collections import deque
    audio_q: deque = deque()

    def audio_cb(indata, n_frames, t, status):
        audio_q.append(indata[:, 0].copy().astype(np.float32))

    # ── 显式双 task：采集发送 + 接收，都由调用方创建并发 ──
    async def capture_loop():
        nonlocal sent_frames   # sent_frames 是闭包外部变量，capture_loop 内自增
        turn_frames_0 = 0      # 本轮分句起始的帧计数
        last_turn_no = 0
        last_frame_at = 0.0
        audio_buf = np.zeros(0, dtype=np.float32)
        t0 = time.monotonic()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=FRAME_SAMPLES, callback=audio_cb):
            while (time.monotonic() - t0) < duration_s and not client.closed:
                while audio_q and len(audio_buf) < CHUNK_SAMPLES:
                    audio_buf = np.concatenate([audio_buf, audio_q.popleft()])
                if len(audio_buf) < CHUNK_SAMPLES:
                    await asyncio.sleep(0.02)
                    continue
                chunk, audio_buf = audio_buf[:CHUNK_SAMPLES], audio_buf[CHUNK_SAMPLES:]

                # 新一轮开始 → 重置本分句的视频帧预算（同回放：预算按分句算）
                if client.turn_no != last_turn_no:
                    last_turn_no = client.turn_no
                    turn_frames_0 = sent_frames

                frame_b64 = []
                now = time.monotonic()
                if cap and fps > 0 and (now - last_frame_at) >= max(1.0 / fps, 0.1) and \
                   (sent_frames - turn_frames_0) * TOKENS_PER_FRAME < kv_budget_tokens:
                    ret, frame = cap.read()
                    if ret:
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frame_b64 = [base64.b64encode(jpeg.tobytes()).decode()]
                        sent_frames += 1
                        last_frame_at = now

                await client.send_input(audio_b64=b64(chunk), video_frames=frame_b64,
                                        force_listen=(turn_decision == "vad_turnsense"))

        # 尾静音：同回放，让 VAD 闭合最后一句并收完还在路上的回复。
        if not client.closed:
            silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            for _ in range(max(0, int(round(tail_silence_s * 1000 / CHUNK_MS)))):
                if client.closed:
                    break
                await client.send_input(
                    audio_b64=b64(silence),
                    force_listen=(turn_decision == "vad_turnsense"))
                await asyncio.sleep(CHUNK_MS / 1000.0)

    cap_task = asyncio.create_task(capture_loop())
    recv_task = asyncio.create_task(client.receive_loop())
    await cap_task
    if not client.closed:
        print(f"\n  [收尾] 采集结束，等待剩余回复"
              f"（{drain_idle_s:.0f}s 无事件即结束）...", flush=True)
        await client.drain(idle_s=drain_idle_s)
    if not recv_task.done():
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass

    if cap:
        cap.release()
    return client.metrics


# ============================================================================
# main
# ============================================================================

def _ssl_ctx_noverify():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ============================================================================
# 并发测试（--concurrency N）
# ============================================================================

def _gpu_stats(gpu_ids=(0, 1)):
    """查询多张 GPU 的显存(used)/利用率。返回 {gpu_id: (used_MiB, util%)}。"""
    ids = ",".join(str(g) for g in gpu_ids)
    try:
        # nvidia-smi 在子进程里 NVML 初始化可能较慢（~7s），timeout 给足；
        # 失败时静默返回 0，不影响主流程。
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", ids],
            capture_output=True, text=True, timeout=15,
        )
        out = {}
        for i, line in enumerate(r.stdout.strip().splitlines()):
            if gpu_ids[i] is not None and line:
                parts = line.split(", ")
                try:
                    out[gpu_ids[i]] = (int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    out[gpu_ids[i]] = (0, 0)
        return out
    except Exception:
        return {g: (0, 0) for g in gpu_ids}


def _gpu_baseline(gpu_ids=(0, 1)):
    """测试前记录显存基线（模型已加载后）。"""
    stats = _gpu_stats(gpu_ids)
    return {g: stats.get(g, (0, 0))[0] for g in gpu_ids}


def _print_summary(metrics: List[TurnMetrics]) -> None:
    """单路结果汇总：平均 TTFT / 平均 speed / 轮数。"""
    ttfts = [m.ttft_s for m in metrics if m.ttft_s is not None]
    speeds = [m.speed_cps for m in metrics if m.speed_cps > 0]
    print("\n=== 汇总 ===")
    if ttfts:
        print(f"  平均首字延迟: {sum(ttfts)/len(ttfts):.2f}s (min={min(ttfts):.2f}s)")
    if speeds:
        print(f"  平均生成速度: {sum(speeds)/len(speeds):.0f}ch/s")
    print(f"  对话轮数: {len(metrics)}")


def _turn_stats(metrics: List[TurnMetrics]) -> dict:
    """从一轮组 metrics 汇总每路指标：轮数、平均/最差 TTFT、平均 speed、总字符。"""
    ttfts = [m.ttft_s for m in metrics if m.ttft_s is not None]
    speeds = [m.speed_cps for m in metrics if m.speed_cps > 0]
    return {
        "n_turns": len(metrics),
        "avg_ttft_s": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "avg_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "chars": sum(m.reply_chars for m in metrics),
    }


async def _run_one_concurrent(url, ssl_ctx, direct, system_prompt,
                              vad_cfg, ts_cfg, video_path, prompt,
                              audio_chunk_ms=1000, kv_budget=20000,
                              max_audio_s=None, tail_silence_s=2.0,
                              drain_idle_s=5.0, accumulate_context=False,
                              echo=False):
    """单路并发会话：独立连接，与单路 run_file_replay 完全相同的发送/收尾/统计。

    echo=False：并发下不逐字流式（多路交错会错乱），只收指标。
    """
    t0 = time.perf_counter()
    client = None
    try:
        client = StreamingChatClient(url, ssl_ctx=ssl_ctx, direct=direct, echo=echo)
        await client.connect()
        await client.init(
            mode="full_duplex",
            system_prompt=system_prompt,
            config={"accumulate_context": accumulate_context, "vad": vad_cfg, "turnsense": ts_cfg},
        )
        setup_ms = (time.perf_counter() - t0) * 1000
        # 与单路完全一致：turn_decision 由 run_file_replay 内部按 backend 推导，
        # 完整音轨连续流式发送 + 同样的收尾/统计。
        metrics = await run_file_replay(
            client, video_path, prompt,
            audio_chunk_ms=audio_chunk_ms,
            kv_budget_tokens=kv_budget,
            max_audio_s=max_audio_s,
            tail_silence_s=tail_silence_s,
            drain_idle_s=drain_idle_s,
        )
        st = _turn_stats(metrics)
        return {
            "ok": bool(metrics) and any(m.reply_chars for m in metrics),
            "setup_ms": setup_ms,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "metrics": metrics,
            "n_turns": st["n_turns"],
            "avg_ttft_s": st["avg_ttft_s"],
            "avg_speed": st["avg_speed"],
            "chars": st["chars"],
        }
    except Exception as e:
        return {"ok": False, "setup_ms": 0, "total_ms": 0, "metrics": [],
                "n_turns": 0, "avg_ttft_s": 0.0, "avg_speed": 0.0,
                "chars": 0, "info": str(e)[:80]}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


def _percentile(vals: List[float], pct: float) -> float:
    """线性插值百分位。"""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def run_concurrency(url, ssl_ctx, direct, system_prompt,
                          vad_cfg, ts_cfg, video_path, prompt,
                          concurrency, max_audio_s, gpu_ids=None,
                          accumulate_context=False,
                          audio_chunk_ms=1000, kv_budget=20000,
                          tail_silence_s=2.0, drain_idle_s=5.0,
                          show_reply_text=True):
    """视频/音频回放的统一入口：N 路各自独立连接，与单路同口径统计。

    concurrency==1 即单路（--concurrency 默认 1）：每路 echo=True 流式输出，
    结束后打单路汇总；N>1 加 GPU 监控 + 每路逐轮汇总表 + 全体 P50/P90/P99。
    gpu_ids=None 或 N==1 时不启 GPU 监控。
    """
    N = concurrency
    gpu_ids = gpu_ids or ()
    multi = N > 1
    print(f"\n=== 回放测试（{N} 路）===")
    print(f"  视频: {video_path} | 每路音频完整音轨（或 --max-audio-s 截断）")

    monitor_running = multi and bool(gpu_ids)
    peak = {g: (0, 0) for g in gpu_ids}
    baseline = {}

    if monitor_running:
        baseline = _gpu_baseline(gpu_ids)
        print(f"  GPU 监控: {list(gpu_ids)} | 显存基线: "
              + ", ".join(f"GPU{g}={v}MiB" for g, v in baseline.items()))

        async def monitor():
            while monitor_running:
                stats = _gpu_stats(gpu_ids)
                for g in gpu_ids:
                    v, u = stats.get(g, (0, 0))
                    if v > peak[g][0]:
                        peak[g] = (v, max(peak[g][1], u))
                    if u > peak[g][1]:
                        peak[g] = (v, u)
                await asyncio.sleep(0.3)

        mon_task = asyncio.create_task(monitor())

    t_all = time.perf_counter()
    tasks = [_run_one_concurrent(
        url, ssl_ctx, direct, system_prompt,
        vad_cfg, ts_cfg, video_path, prompt,
        audio_chunk_ms=audio_chunk_ms, kv_budget=kv_budget,
        max_audio_s=max_audio_s, tail_silence_s=tail_silence_s,
        drain_idle_s=drain_idle_s, accumulate_context=accumulate_context,
        echo=not multi,   # 单路可流式；多路不流式，避免交错
    ) for _ in range(N)]
    results = await asyncio.gather(*tasks)

    if monitor_running:
        monitor_running = False
        await mon_task
    wall_ms = (time.perf_counter() - t_all) * 1000

    ok = sum(1 for r in results if r["ok"])

    if not multi:
        # 单路：echo 已流式打印每轮 summary，这里只打汇总
        _print_summary(results[0]["metrics"])
        print(f"  wall={wall_ms:.0f}ms")
        return ok

    print(f"\n  --- {N} 路并发 ---")
    for i, r in enumerate(results):
        st = "PASS" if r["ok"] else "FAIL"
        extra = f" {r.get('info','')}" if not r["ok"] else ""
        print(f"    #{i+1:<2} {st:<4} setup={r['setup_ms']:.0f}ms "
              f"{r['n_turns']}轮 平均TTFT={r['avg_ttft_s']:.2f}s "
              f"平均speed={r['avg_speed']:.0f}ch/s 总{r['chars']}ch{extra}")
        for m in r["metrics"]:
            print(f"      {m.summary}")
            if show_reply_text and m.reply_text:
                print(f"        {m.reply_text}")

    if monitor_running:
        peak_str = ", ".join(
            f"GPU{g}: {v}MiB/{u}% (基线{baseline.get(g,0)}MiB)" for g, v, u in
            [(g, p[0], p[1]) for g, p in peak.items()])
        print(f"  峰值显存/利用率: {peak_str}")

    # 全体统计：所有路的每轮 TTFT / speed 聚合出平均 + P50/P90/P99
    all_ttfts = [m.ttft_s for r in results for m in r["metrics"] if m.ttft_s is not None]
    all_speeds = [m.speed_cps for r in results for m in r["metrics"] if m.speed_cps > 0]
    print(f"  全体: 平均TTFT={sum(all_ttfts)/len(all_ttfts):.2f}s "
          f"P50={_percentile(all_ttfts, 0.5):.2f}s P90={_percentile(all_ttfts, 0.9):.2f}s "
          f"P99={_percentile(all_ttfts, 0.99):.2f}s | "
          f"平均speed={sum(all_speeds)/len(all_speeds):.0f}ch/s "
          f"P50={_percentile(all_speeds, 0.5):.0f} P90={_percentile(all_speeds, 0.9):.0f} "
          f"P99={_percentile(all_speeds, 0.99):.0f}" if all_ttfts and all_speeds else "")
    print(f"  {ok}/{N} PASS | wall={wall_ms:.0f}ms")

    print(f"\n=== 并发测试完成: {ok}/{N} 路成功 ===")
    return ok


async def main():
    parser = argparse.ArgumentParser(description="流式视频对话 Demo")
    parser.add_argument("--direct-backend", default="",
                        help="直连后端 WS 地址(如 ws://127.0.0.1:22500/backend)，绕过 gateway")
    parser.add_argument("--gateway", default="wss://127.0.0.1:8006/v1/realtime",
                        help="gateway WS 地址")
    parser.add_argument("--host", default="192.168.89.106",
                        help="服务主机地址（内网机器连生产用，如 192.168.89.106）")
    parser.add_argument("--port", type=int, default=8006,
                        help="服务端口：8006=gateway(wss，生产外网入口，默认)；"
                             "22500=直连后端(ws，需后端端口映射到宿主机)。"
                             "未指定时走 --direct-backend / --gateway 默认地址")
    
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发路数(默认1=单路)。>1 时进入并发测试，同时监控 GPU 显存/利用率")
    parser.add_argument("--gpu-ids", default="0,1",
                        help="要监控的 GPU 编号(逗号分隔)，默认 '0,1'")
    parser.add_argument("--no-reply-text", action="store_true",
                        help="并发结果不打印每轮完整回复文本（只看指标）")
    parser.add_argument("--realtime", action="store_true", help="实时采集模式（麦克风+摄像头）")
    parser.add_argument("--duration", type=float, default=30.0, help="实时模式时长(秒)")
    
    parser.add_argument("--video", default="assets/video/turnbased/121.mp4", help="视频文件路径（回放模式）")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="视频抽帧率(帧/秒)，0=不抽帧只发音频") 
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最大抽帧数，0=不限(用 KV 预算保护)")
    parser.add_argument("--audio", default="",
                        help="音频文件：与 --video 同时用=替换视频音轨；单独用=纯音频交互模式")
    parser.add_argument("--audio-chunk-ms", type=int, default=1000,
                        help="demo 发送音频的 chunk 大小(ms)，默认1000（VAD 需 ≥25ms，"
                             "100 更实时但更频繁）。回放模式即用此值")   
    parser.add_argument("--max-audio-s", type=float, default=None,
                        help="限制音频提取时长(秒)，默认=完整音轨。长音频约25 tok/s，注意 KV")
    parser.add_argument("--kv-budget", type=int, default=20000,
                        help="KV 预算(tokens)，每帧≈540，超过后停止发帧保留音频。"
                             "默认 20000 ≈ 生产每路 KV(25600) 的 78%，留余量给音频+历史")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="回放速度倍率，1.0=真实速度(1s音频等1s)，>1 加速发送")
    parser.add_argument("--tail-silence-s", type=float, default=2.0,
                        help="音频发完后补发的静音时长(秒)，默认2.0。FSMN VAD 靠尾静音"
                             "闭合语音段，不补则最后一句可能永远不触发回复")
    parser.add_argument("--drain-idle-s", type=float, default=5.0,
                        help="收尾空闲兜底(秒)，默认5。正常路径不走这里——服务端每个 push 都回"
                             "带递增 response_id 的回执，追平即知输入已消费完并立即退出；"
                             "只有序号对不上时(如 --wait-reply=false)才退回空闲判据")
    parser.add_argument("--wait-reply", type=lambda v: v.lower() in ("1","true","yes","on"),
                        default=True,
                        help="VAD+TurnSense 触发回复后是否暂停发送音频，等模型回复完成再继续(默认开)。"
                             "--wait-reply=false 关闭则持续发送不等回复")
    
    parser.add_argument("--system-prompt", default="你是一个友好的中文助手。",
                        help="系统提示词")
    parser.add_argument("--prompt", default="你是一个多模态AI助手，请理解音频和视频内容，简洁准确地回复用户。",
                        help="对话提示词")
    parser.add_argument("--accumulate-context", action="store_true",
                        help="跨分句累积历史上下文（默认关）。开启后每轮保留之前的音频/回复，"
                             "模型有跨轮记忆，但 prompt 变长、输出变慢；关闭则每轮只推理当前分句，输出更快")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="单次回复最大生成 token 数(默认1024)。长回复(视频理解/总结)易超 512，"
                             "默认后端 1024；过大则回复慢、KV 消耗大")
    
    parser.add_argument("--vad-model", choices=["fsmn", "silero"], default="fsmn",
                        help="VAD 模型: fsmn(FunASR,默认) 或 silero")
    parser.add_argument("--vad-tail-sil", type=int, default=600,
                        help="VAD 尾静音(ms)，fsmn 的 max_end_silence_time，默认600")
    parser.add_argument("--vad-max-len", type=int, default=30000,
                        help="VAD 最大段长(ms)，fsmn 的 max_single_segment_time，默认30000")
    parser.add_argument("--ts-wait-ms", type=int, default=1000,
                        help="TurnSense incomplete 等待(ms)：语义不完整时等多久，超时强制回复，默认1000")
    parser.add_argument("--ts-invalid-threshold", type=float, default=0.5,
                        help="TurnSense invalid 丢弃阈值(0~1)：invalid 概率≥此值则丢弃该句不回复，默认0.5")


    args = parser.parse_args()

    if not args.video and not args.audio and not args.realtime and args.concurrency <= 1:
        parser.error("需要 --video / --audio 或 --realtime 之一（纯音频模式用 --audio）")

    if args.concurrency > 1 and not args.video:
        parser.error("并发测试需要 --video 提供测试音视频")

    # 按 --audio-chunk-ms 重算全局发送 chunk（并发/实时模式用；回放模式直接传参）
    global CHUNK_MS, CHUNK_SAMPLES
    CHUNK_MS = max(25, args.audio_chunk_ms)
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

    # 连接目标：直连后端 或 gateway。
    # 优先级：--host+--port 拼接（显式指定时）> --direct-backend/--gateway 显式地址 > 默认。
    # 默认走 gateway（--gateway 默认 wss://127.0.0.1:8006），与生产外网入口一致。
    ssl_ctx = None
    direct = bool(args.direct_backend)
    if args.port > 0 or args.host:
        # --host/--port 拼接生产服务地址：内网其他机器可连
        if args.port == 22500:
            # 后端直连（容器内 WS，需后端端口映射到宿主机）
            url = f"ws://{args.host or '127.0.0.1'}:{args.port}/backend"
            direct = True
        else:
            # gateway（HTTPS/WSS）；默认端口 8006
            url = f"wss://{args.host or '127.0.0.1'}:{args.port or 8006}/v1/realtime"
            direct = False
    else:
        url = args.direct_backend or args.gateway
        direct = bool(args.direct_backend)
    if not direct:
        # gateway 是 HTTPS/WSS，需要 no-verify + mode 参数
        ssl_ctx = _ssl_ctx_noverify()
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}mode=video"
    elif url.startswith("wss"):
        ssl_ctx = _ssl_ctx_noverify()

    print(f"连接目标: {url} (direct={direct})")

    # ── 实时采集：麦克风+摄像头，只支持单路（无法并发），独立路径 ──
    if args.realtime:
        client = StreamingChatClient(url, ssl_ctx=ssl_ctx, direct=direct)
        try:
            await client.connect()
            await client.init(
                mode="full_duplex",
                system_prompt=args.system_prompt,
                config={
                    "accumulate_context": args.accumulate_context,
                    "vad": {
                        "vad_model": args.vad_model,
                        "vad_tail_sil": args.vad_tail_sil,
                        "vad_max_len": args.vad_max_len,
                    },
                    "turnsense": {
                        "incomplete_wait_ms": args.ts_wait_ms,
                        "invalid_confidence_threshold": args.ts_invalid_threshold,
                    },
                },
            )
            turn_decision = "vad_turnsense" if client.backend == "qwen3omni" else "model"
            print(f"会话已建立: backend={client.backend} turn_decision={turn_decision}")
            metrics = await run_realtime(client, args.system_prompt,
                                         turn_decision, args.duration,
                                         fps=args.fps, kv_budget_tokens=args.kv_budget,
                                         tail_silence_s=args.tail_silence_s,
                                         drain_idle_s=args.drain_idle_s)
            if metrics:
                _print_summary(metrics)
        finally:
            await client.close()
        return

    # ── 视频/音频回放：单路(--concurrency 1)与并发(N>1)统一走 run_concurrency ──
    # 每路独立连接 + 完整音轨流式，与单路完全相同的发送/收尾/指标口径。
    gpu_ids = (tuple(int(x.strip()) for x in args.gpu_ids.split(",") if x.strip())
               if args.concurrency > 1 else None)
    await run_concurrency(
        url, ssl_ctx, direct, args.system_prompt,
        vad_cfg={
            "vad_model": args.vad_model,
            "vad_tail_sil": args.vad_tail_sil,
            "vad_max_len": args.vad_max_len,
        },
        ts_cfg={
            "incomplete_wait_ms": args.ts_wait_ms,
            "invalid_confidence_threshold": args.ts_invalid_threshold,
        },
        video_path=args.video,
        prompt=args.prompt,
        concurrency=args.concurrency,
        max_audio_s=args.max_audio_s,
        gpu_ids=gpu_ids,
        accumulate_context=args.accumulate_context,
        audio_chunk_ms=args.audio_chunk_ms,
        kv_budget=args.kv_budget,
        tail_silence_s=args.tail_silence_s,
        drain_idle_s=args.drain_idle_s,
        show_reply_text=not args.no_reply_text,
    )


if __name__ == "__main__":
    asyncio.run(main())
