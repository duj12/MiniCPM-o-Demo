#!/usr/bin/env python3
"""流式视频对话 Demo — 支持音视频文件回放 / 实时采集，两种 turn_decision 判决。

复用 MiniCPM-o-Demo 的全双工流式协议（/v1/realtime?mode=video）：
  - 音视频持续逐帧发送（"边听边看"，累积进 KV cache）
  - turn_decision="model"        : 模型自主决定 speak/listen（现有全双工）
  - turn_decision="vad_turnsense": VAD 检测语音停顿 + TurnSense 语义完整后才回复
    （worker 端用 force_listen 累积，说完才触发 decode —— 首字延迟更低）

用法：
  # 单路回放（视频理解，VAD+TurnSense 判决；音频 100ms 细粒度流式）
  python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
      --turn-decision vad_turnsense --prompt "你是一个多模态助手，请简练回复用户的问题。" \
      --backend qwen3omni --direct-backend ws://127.0.0.1:22500/backend

  # 并发压力测试（10 路，用完整音轨驱动真实多轮；GPU 显存/利用率监控）
  python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
      --turn-decision vad_turnsense --prompt "你是一个多模态助手，请简练回复用户的问题。" \
      --backend qwen3omni --direct-backend ws://127.0.0.1:22500/backend \
      --concurrency 10 --gpu-ids 0,1

  # 实时采集（麦克风 + 摄像头）。模型自主判决仅 MiniCPM 支持：
  python streaming_chat_demo.py --realtime --turn-decision model --backend minicpm
  # qwen3omni 实时采集用 VAD+TurnSense：
  python streaming_chat_demo.py --realtime --turn-decision vad_turnsense --backend qwen3omni

  # 走 gateway（生产路径，wss + mode=video；需可访问 :8006）
  python streaming_chat_demo.py --video xxx.mp4 --turn-decision vad_turnsense --backend qwen3omni

  # 内网其他机器连生产服务（--host/--port 拼接地址）：
  #   --port 22500 = 直连后端(ws://HOST:22500/backend)；--port 8006 = gateway(wss://HOST:8006/v1/realtime)
  python streaming_chat_demo.py --video xxx.mp4 --turn-decision vad_turnsense \
      --backend qwen3omni --host 192.168.89.106 --port 22500

常用参数：
  --audio-chunk-ms  音频发送块大小(ms)，默认 100（VAD 需 ≥25ms；100ms 兼顾实时性）
  --kv-budget       单分句视频帧预算(token)，默认 20000（≈每路 KV 25600 的 78%）
  --max-audio-s     限制回放/并发使用的音频时长(秒)，默认全部
  --concurrency N   并发路数，>0 进入并发测试
  --host / --port   生产服务地址（内网机器连用；22500=后端，8006=gateway）
  --direct-backend  直连后端 WS（绕过 gateway），如 ws://127.0.0.1:22500/backend

输出指标（每轮）：
  - TTFT      : 首字延迟（发送完触发帧 → 第一个文本 token）
  - in_audio_s: 本轮输入音频长度（秒）
  - in_video_s: 本轮输入视频长度（秒）
  - reply_s   : 回复总耗时
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
from typing import List, Optional

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

    @property
    def summary(self) -> str:
        ttft_str = f"{self.ttft_s:.2f}s" if self.ttft_s is not None else "N/A"
        return (
            f"turn#{self.turn_idx}: TTFT={ttft_str} "
            f"in_audio={self.in_audio_s:.1f}s in_video={self.in_video_s:.1f}s "
            f"reply={self.reply_s:.1f}s ({self.reply_chars}ch, "
            f"{self.speed_cps:.0f}ch/s) state={self.model_state}"
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
                 direct: bool = False, backend: str = "qwen3omni"):
        self.url = url
        self.ssl_ctx = ssl_ctx
        self.max_size = max_size
        self.direct = direct
        # 后端类型：minicpm（free-duplex，回复以 listen 结束，response.done 只是 chunk
        # 边界）或 qwen3omni（turn-based，response.done 即完整回复结束）。
        self.backend = backend
        self.ws = None
        self.session_id = None

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
                   turn_decision: str = "model", use_tts: bool = False,
                   config: Optional[dict] = None) -> dict:
        """发送 session.init。turn_decision 决定用模型自主 还是 VAD+TurnSense 判决。"""
        payload = {
            "mode": mode,
            "use_tts": use_tts,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        cfg = dict(config or {})
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
                         text: str = "", force_listen: bool = False) -> None:
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
        await self.ws.send(json.dumps({"type": "input.append", "input": inp}, ensure_ascii=False))

    async def collect_reply(self, timeout: float = 300.0,
                            listen_giveup_s: float = 8.0) -> TurnMetrics:
        """收集一回合的回复（text/audio/listen deltas + response.done）。

        返回指标：TTFT（首个文本 delta）、回复耗时、字符数、模型状态。

        - timeout: 单次 recv 超时（总体等待上限）
        - listen_giveup_s: 若模型持续 listen（选择聆听）超过该秒数，判定本轮无回复
          （避免对静音/短音频时 collect_reply 无限等待 response.done）
        """
        import websockets
        m = TurnMetrics()
        t_start = time.monotonic()
        first_delta_at = None
        reply_started = None
        last_listen_at = None
        text = ""
        done = False

        while not done:
            # 若收到过 listen（模型选择聆听）且之后一段时间无任何新事件，判定无回复
            if last_listen_at is not None and (time.monotonic() - last_listen_at) > listen_giveup_s:
                break
            try:
                ev = await self._recv(min(timeout, listen_giveup_s))
            except (TimeoutError, websockets.ConnectionClosed):
                # 无新事件：若已进入聆听或等待过久，结束本轮
                break

            t = ev.get("type")
            if t == "response.output.delta":
                kind = ev.get("kind")
                if kind == "text":
                    chunk = ev.get("text", "")
                    if chunk:
                        if first_delta_at is None:
                            first_delta_at = time.monotonic()
                            reply_started = first_delta_at
                            m.ttft_s = first_delta_at - t_start
                        text += chunk
                        # 不在此逐 token 打印（碎片化难看）；回复完成后统一显示
                elif kind == "listen":
                    m.model_state = "listening"
                    if first_delta_at is None:
                        # 还没收到任何文本就 listen：模型选择聆听（无回复）
                        last_listen_at = time.monotonic()
                        m.model_state = "listening"
                    else:
                        # 已收到文本后 listen = 本轮回复结束。
                        # MiniCPM free-duplex 多 chunk 回复以 listen 结束（而非
                        # response.done，后者只是 chunk 边界）。
                        done = True
                elif kind == "audio":
                    m.model_state = "speaking"
            elif t == "response.done":
                # Qwen3-Omni turn-based：response.done = 完整回复结束。
                # MiniCPM free-duplex：response.done 只是 chunk 边界（其 text 已通过
                # text delta 累积），listen 才是 turn 结束。
                if self.backend != "minicpm":
                    text = ev.get("text", "") or text
                    # Qwen3 收到 response.done 即本轮有完整文本回复 → speaking。
                    # 此前若从未收到 listen/audio，model_state 会留空，这里补齐。
                    if not m.model_state:
                        m.model_state = "speaking"
                    done = True
            elif t == "session.closed":
                break
            elif t == "session.error":
                print(f"  [error] {ev}")
                break

        m.reply_chars = len(text)
        # reply_s: 从首个文本 delta 到结束（纯生成时间），避免把触发前的
        # 等待算进去导致 speed 虚高。
        if reply_started is not None:
            m.reply_s = (time.monotonic() - reply_started)
            m.speed_cps = m.reply_chars / m.reply_s if m.reply_s > 0 else 0.0
        else:
            m.reply_s = (time.monotonic() - t_start)
            m.speed_cps = 0.0
        m.reply_text = text
        return m

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
                          prompt: str, turn_decision: str, n_turns: int = 1,
                          fps: float = 1.0, max_frames: int = 0,
                          kv_budget_tokens: int = 20000,
                          audio_path: str = "",
                          max_audio_s: Optional[float] = None,
                          replay_speed: float = 1.0,
                          wait_reply: bool = True,
                          audio_chunk_ms: int = 100):
    """离线：按 fps 流式抽帧 + 音频切块发送，模拟实时流。

    帧策略：按 fps 持续抽帧（max_frames=0 不限制总数），但用 KV 预算保护
    ——每帧 ≈ 540 token（缩放后），累计超过 kv_budget_tokens 后停止发帧
    （音频继续），避免撑爆 KV cache。这样"最新画面"始终在流，旧帧滚动丢弃。

    audio_path: 指定人声音频 wav（替代视频音轨）。视频音轨常为非人声背景音，
      TurnSense 会判 invalid；用清晰人声可验证 VAD+TurnSense 完整链路。
    max_audio_s: 限制音频提取时长（秒）。None=完整音轨。注意长音频 token
      量巨大（约 25 tok/s），可能超出 KV，需配合 --kv-budget / 后端 -c。
    """
    # 回放模式音频块节奏：默认 100ms（VAD 需 ≥25ms chunk，100ms 兼顾实时性与网络
    # 开销），可由 --audio-chunk-ms 覆盖。视频帧节奏与音频解耦：每
    # _frames_per_audio_chunk 个音频块（1s）发 1 帧，保持视频 1s 一帧不爆 KV。
    # 并发/实时模式仍用全局 CHUNK_MS（由 --audio-chunk-ms 控制）。
    _chunk_ms = max(25, audio_chunk_ms)
    _chunk_samples = SAMPLE_RATE * _chunk_ms // 1000
    _frames_per_audio_chunk = max(1, 1000 // _chunk_ms)  # 默认 10 → 每 1s 发 1 帧

    audio = extract_audio_pcm(video_path, max_s=max_audio_s)
    if audio_path and os.path.exists(audio_path):
        if sf is None:
            print("  [warn] soundfile 不可用，忽略 --audio")
        else:
            a, sr = sf.read(audio_path, dtype="float32")
            if sr != SAMPLE_RATE:
                print(f"  [warn] --audio 采样率 {sr}≠16000，将按原采样率处理（可能不准）")
            audio = np.asarray(a, dtype=np.float32)
    frames = extract_frames_evenly(video_path, fps=fps, max_frames=max_frames)
    video_s = probe_duration(video_path)
    audio_s = len(audio) / SAMPLE_RATE

    print(f"\n=== 文件回放: {os.path.basename(video_path)} ===")
    print(f"  音频 {audio_s:.1f}s, 视频 {video_s:.1f}s, 抽帧 {len(frames)} 帧 @{fps}fps")
    print(f"  turn_decision = {turn_decision}, KV预算 ≈{kv_budget_tokens} tok")

    TOKENS_PER_FRAME = 540  # 1440p 缩放后每帧约 540 token

    # ── VAD 驱动 + wait_reply ──
    # 不硬切段：持续流式发送音频，worker 的 VAD 检测到静音停顿即视为一个语音段，
    # 触发 TurnSense → 回复。demo 每发一块就检查是否已触发回复；若 wait_reply 开启，
    # 触发后暂停发送，等模型回复完成再继续（避免后续音频淹没回复内容）。
    all_metrics = []
    turn_no = 0
    sent_frames = 0
    n_chunks = max(1, int(np.ceil(len(audio) / _chunk_samples)))
    i = 0
    in_reply = False
    rm = TurnMetrics(turn_idx=0)
    turn_start_i = 0         # 本轮分句开始时的 chunk 索引（用于计算当前分句音频时长）
    cur_turn_audio_s = 0.0   # 当前分句的音频时长（触发时计算，回复完成后用于指标）
    turn_start_frames = 0    # 本轮分句开始时的累计发送帧数（用于计算当前分句视频时长）
    cur_turn_video_s = 0.0   # 当前分句的视频时长（触发时计算，回复完成后用于指标）

    while i < n_chunks:
        if not in_reply:
            chunk = audio[i * _chunk_samples:(i + 1) * _chunk_samples]
            if len(chunk) < _chunk_samples:
                chunk = np.pad(chunk, (0, _chunk_samples - len(chunk)))

            frame_b64 = []
            # 视频帧节奏独立于音频：每 _frames_per_audio_chunk 个音频块（1s）发 1 帧。
            # 这样音频可 100ms 细粒度流式（VAD 更实时），而视频保持 1s 一帧不爆 KV。
            # kv_budget 限制的是"当前分句"发送的视频帧数（单次持续处理的帧预算），
            # 不是累计发送帧数——否则多轮对话后预算被跨分句耗尽，视频帧提前停发。
            if i % _frames_per_audio_chunk == 0 and \
               (sent_frames - turn_start_frames) * TOKENS_PER_FRAME < kv_budget_tokens and frames:
                frame_b64 = [b64(frames[sent_frames % len(frames)])]
                sent_frames += 1

            text = prompt if i == 0 else ""
            # force_listen 只用于 VAD+TurnSense 判决（worker 累积音频，VAD 触发才回复）。
            # 自由全双工（turn_decision=model）下必须让模型自主 listen/speak，
            # 不能 force_listen，否则模型永远只累积不回复。
            await client.send_input(
                audio_b64=b64(chunk), video_frames=frame_b64, text=text,
                force_listen=(turn_decision == "vad_turnsense"),
            )
            i += 1
            await asyncio.sleep(_chunk_ms / 1000.0 / replay_speed)

        # 非阻塞检查：仅 VAD+TurnSense 模式触发回复后暂停发送。
        # 自由全双工（turn_decision=model）靠持续输入驱动，模型可随时回复，
        # 不能暂停发送（否则模型收不到后续输入，回复会中断）。
        if turn_decision == "vad_turnsense" and wait_reply and not in_reply:
            try:
                rm = await asyncio.wait_for(
                    client.collect_reply(timeout=0.4, listen_giveup_s=0.4),
                    timeout=0.5,
                )
                if rm.reply_chars > 0 or rm.model_state == "speaking":
                    in_reply = True
                    turn_no += 1
                    rm.turn_idx = turn_no
                    # 当前分句音频时长 = 从上一轮触发后（turn_start_i）到本次触发（i）
                    # 之间发送的音频块数 × 每块秒数。触发后记录下一分句起点。
                    cur_turn_audio_s = (i - turn_start_i) * _chunk_ms / 1000.0
                    turn_start_i = i
                    # 当前分句视频时长 = 从上一轮触发后到本次触发之间发送的帧数 / fps。
                    # 帧发送受 kv_budget_tokens 限制，可能比音频稀疏，但秒数估算仍准确。
                    cur_turn_video_s = (sent_frames - turn_start_frames) / max(fps, 0.1)
                    turn_start_frames = sent_frames
                    print(f"\n  [VAD段{turn_no}] 模型开始回复，暂停发送...")
            except (asyncio.TimeoutError, TimeoutError):
                pass

        # 回复进行中：仅 VAD+TurnSense 模式等待回复完成再继续发送。
        # model 模式持续发送，回复由模型自主穿插（靠 listen_giveup_s 分段）。
        if turn_decision == "vad_turnsense" and in_reply:
            # 等待回复完成。不在这里发 keep-alive：静音 prefill 会让模型
            # 持续 Listen，导致本循环永远收不到 text 而卡住。后端 reaper
            # 超时已调大（见 server-qwen3omni 的 idle_timeout），长回复不会
            # 被回收。
            rm_done = await client.collect_reply(timeout=30.0, listen_giveup_s=15.0)
            if rm_done.reply_chars > 0 or rm_done.model_state == "speaking":
                rm = rm_done
                rm.turn_idx = turn_no
            # 回复完成（response.done 或 listen 超时）→ 记录，恢复发送
            # in_audio_s / in_video_s 用当前分句的时长（VAD 触发的那一段），
            # 不是整段音轨/视频的累计值。
            m = TurnMetrics(turn_idx=turn_no, in_audio_s=cur_turn_audio_s)
            m.in_video_s = cur_turn_video_s
            m.ttft_s = rm.ttft_s
            m.reply_s = rm.reply_s
            m.reply_chars = rm.reply_chars
            m.speed_cps = rm.speed_cps
            m.model_state = rm.model_state
            m.reply_text = rm.reply_text
            all_metrics.append(m)
            print(f"\n  {m.summary}")
            if m.reply_text:
                print(f"  ── 回复{turn_no} ──\n  {m.reply_text}\n  ──────────────")
            in_reply = False
            await asyncio.sleep(0.5)  # 短暂停顿后继续发下一段

    return all_metrics


# ============================================================================
# 实时采集模式
# ============================================================================

async def run_realtime(client: StreamingChatClient, system_prompt: str,
                       turn_decision: str, duration_s: float = 30.0,
                       fps: float = 1.0, kv_budget_tokens: int = 20000):
    """在线：麦克风 + 摄像头实时采集。需要 sounddevice + opencv。

    视频按 fps 抽帧（KV 预算保护：接近上限后只发音频，保留最新画面）。
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

    frame = None
    cap = cv2.VideoCapture(0) if cv2 else None
    if cap and not cap.isOpened():
        cap = None

    loop = asyncio.get_event_loop()
    all_metrics = []
    m = TurnMetrics(turn_idx=1)
    t_session_start = time.monotonic()
    cur_audio = 0.0
    cur_frames = 0

    audio_buf = np.zeros(0, dtype=np.float32)
    stop = False

    def audio_cb(indata, frames, t, status):
        nonlocal audio_buf
        audio_buf = np.concatenate([audio_buf, indata[:, 0].astype(np.float32)])

    TOKENS_PER_FRAME = 540
    frame_tokens = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SAMPLES,
                        callback=audio_cb):
        while time.monotonic() - t_session_start < duration_s and not stop:
            # 攒够 1s 音频 → 发送（附当前摄像头帧）
            if len(audio_buf) >= CHUNK_SAMPLES:
                chunk = audio_buf[:CHUNK_SAMPLES]
                audio_buf = audio_buf[CHUNK_SAMPLES:]
                frame_b64 = []
                if cap and fps > 0:
                    # fps 控制：每 (1/fps) 秒抽一帧；KV 预算保护：超限后只发音频
                    now = time.monotonic()
                    if (now - t_session_start) % max(1.0 / fps, 0.1) < 0.15 \
                            and frame_tokens + TOKENS_PER_FRAME < kv_budget_tokens:
                        ret, frame = cap.read()
                        if ret:
                            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_b64 = [base64.b64encode(jpeg.tobytes()).decode()]
                            frame_tokens += TOKENS_PER_FRAME
                            cur_frames += 1
                await client.send_input(audio_b64=b64(chunk), video_frames=frame_b64,
                                        force_listen=(turn_decision == "vad_turnsense"))
                cur_audio += CHUNK_MS / 1000.0   # 块时长（100ms→0.1s）

            # 读取模型回复（非阻塞）
            try:
                rm = await asyncio.wait_for(client.collect_reply(timeout=0.5), timeout=0.6)
            except (asyncio.TimeoutError, TimeoutError):
                continue

            if rm.ttft_s is not None or rm.reply_chars > 0 or rm.model_state:
                m.ttft_s = rm.ttft_s
                m.reply_chars = rm.reply_chars
                m.reply_s = rm.reply_s
                m.speed_cps = rm.speed_cps
                m.model_state = rm.model_state
                m.in_audio_s = cur_audio
                m.in_video_s = cur_frames  # 帧数近似
                all_metrics.append(m)
                print(f"  {m.summary}")
                m = TurnMetrics(turn_idx=len(all_metrics) + 1)
                cur_audio = 0.0
                cur_frames = 0

            await asyncio.sleep(0.05)

    if cap:
        cap.release()
    return all_metrics


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


async def _run_one_concurrent(url, ssl_ctx, direct, backend,
                              turn_decision, system_prompt,
                              vad_cfg, ts_cfg, audio_frames, prompt,
                              n_frames=5):
    """单路并发会话：独立连接，发短音频触发一轮回复，返回结果 dict。"""
    import websockets
    t0 = time.perf_counter()
    client = None
    try:
        client = StreamingChatClient(url, ssl_ctx=ssl_ctx, direct=direct, backend=backend)
        await client.connect()
        await client.init(
            mode="full_duplex",
            system_prompt=system_prompt,
            turn_decision=turn_decision,
            config={"vad": vad_cfg, "turnsense": ts_cfg},
        )
        setup_ms = (time.perf_counter() - t0) * 1000

        text = ""
        first_ts = None
        trigger_ts = None   # VAD+TurnSense 分句决定（收到 turn.turnsense complete）时刻
        t_send = time.perf_counter()  # 发送第一块音频前
        send_end = None

        # 流式发送 + 非阻塞收：每发一块（CHUNK_MS，默认 100ms），用短超时 _recv
        # 收中间事件，再发下一块。不并发 send/recv（websockets 同连接不能同时
        # send+recv），而是交替进行，模拟真实流式（VAD 边收边检测停顿分句）。
        done = False
        had_text = False
        recv_deadline = time.perf_counter() + 45  # 单路总等待上限

        for i, frame in enumerate(audio_frames[:n_frames]):
            if done:
                break
            await client.send_input(
                audio_b64=b64(frame),
                text=prompt if i == 0 else "",
                force_listen=(turn_decision == "vad_turnsense"),
            )
            # 发送间隙非阻塞收事件（每次最多等 0.3s，随后继续发下一块）
            try:
                while True:
                    ev = await asyncio.wait_for(client._recv(timeout=1.0), timeout=1.0)
                    t = ev.get("type")
                    # VAD+TurnSense 分句决定：worker 在 _trigger_reply 时发 complete
                    if t == "turn.turnsense" and ev.get("label") == "complete":
                        if trigger_ts is None:
                            trigger_ts = time.perf_counter()
                    elif t == "response.output.delta":
                        k = ev.get("kind")
                        if k == "text":
                            chunk = ev.get("text", "")
                            if chunk:
                                if first_ts is None:
                                    first_ts = time.perf_counter()
                                text += chunk
                                had_text = True
                        elif k == "listen":
                            if had_text:
                                done = True
                    elif t == "response.done":
                        text = ev.get("text", "") or text
                        if text and first_ts is None:
                            first_ts = time.perf_counter()
                        if text:
                            had_text = True
                        done = True
                    elif t in ("session.closed", "error"):
                        done = True
            except (asyncio.TimeoutError, TimeoutError):
                pass  # 无新事件，继续发下一块

        send_end = time.perf_counter()  # 全部音频发完的时刻

        # 发完后继续收直到 listen（有 text 后）或 done
        while not done and time.perf_counter() < recv_deadline:
            remaining = recv_deadline - time.perf_counter()
            try:
                ev = await client._recv(timeout=min(15.0, max(1.0, remaining)))
            except (TimeoutError, websockets.exceptions.ConnectionClosed):
                break
            t = ev.get("type")
            # VAD+TurnSense 分句决定
            if t == "turn.turnsense" and ev.get("label") == "complete":
                if trigger_ts is None:
                    trigger_ts = time.perf_counter()
            elif t == "response.output.delta":
                k = ev.get("kind")
                if k == "text":
                    chunk = ev.get("text", "")
                    if chunk:
                        if first_ts is None:
                            first_ts = time.perf_counter()
                        text += chunk
                        had_text = True
                elif k == "listen":
                    if had_text:
                        done = True
            elif t == "response.done":
                text = ev.get("text", "") or text
                if text and first_ts is None:
                    first_ts = time.perf_counter()
                if text:
                    had_text = True
                done = True
            elif t in ("session.closed", "error"):
                done = True

        # TTFT：VAD+TurnSense 分句决定（收到 turn.turnsense complete）→ 首字。
        # 用户说完一句话、worker 判定句尾触发回复的时刻开始计时，到模型首字。
        start_ts = trigger_ts or t_send  # 若未收到分句事件，回退到发送开始
        ttft_ms = (first_ts - start_ts) * 1000 if first_ts else 0
        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "ok": bool(text),
            "setup_ms": setup_ms,
            "total_ms": total_ms,
            "ttft_ms": ttft_ms,
            "chars": len(text),
            "info": text[:40] if text else "",
        }
    except Exception as e:
        return {"ok": False, "setup_ms": 0, "total_ms": 0, "ttft_ms": 0,
                "chars": 0, "info": str(e)[:60]}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


async def run_concurrency(url, ssl_ctx, direct, backend,
                          turn_decision, system_prompt,
                          vad_cfg, ts_cfg, video_path, prompt,
                          concurrency, max_audio_s, gpu_ids=(0, 1)):
    """并发 N 路测试：递增并发数，同时监控多张 GPU 显存/利用率。

    每路发同一段音频（完整音轨，或 --max-audio-s 截断），驱动真实多轮。
    """
    import numpy as _np

    # 提取音频：max_audio_s>0 截断到 N 秒，None/0=完整音轨（每路用同一段）
    max_s = float(max_audio_s) if max_audio_s and max_audio_s > 0 else None
    audio = extract_audio_pcm(video_path, max_s=max_s)
    if len(audio) == 0:
        audio = _np.zeros(int(SAMPLE_RATE * (max_audio_s or 3)), dtype=_np.float32)
    # 切成 CHUNK_MS 音频块（默认 100ms；由 --audio-chunk-ms 控制）
    audio_frames = [audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                    for i in range(max(1, len(audio) // CHUNK_SAMPLES))]
    audio_frames = [f if len(f) == CHUNK_SAMPLES
                    else _np.pad(f, (0, CHUNK_SAMPLES - len(f))) for f in audio_frames]

    print(f"\n=== 并发测试 ===")
    print(f"  视频: {video_path} | 每路音频 {len(audio_frames)}s | 目标并发: {concurrency} 路")
    print(f"  GPU 监控: {list(gpu_ids)}")

    baseline = _gpu_baseline(gpu_ids)
    print(f"  显存基线: " + ", ".join(f"GPU{g}={v}MiB" for g, v in baseline.items()))

    N = concurrency
    monitor_running = True
    peak = {g: (0, 0) for g in gpu_ids}  # gpu_id -> (vram, util)

    async def monitor():
        nonlocal peak
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
        url, ssl_ctx, direct, backend, turn_decision, system_prompt,
        vad_cfg, ts_cfg, audio_frames, prompt,
        n_frames=len(audio_frames),
    ) for _ in range(N)]
    results = await asyncio.gather(*tasks)

    monitor_running = False
    await mon_task
    wall_ms = (time.perf_counter() - t_all) * 1000

    ok = sum(1 for r in results if r["ok"])

    print(f"\n  --- {N} 路并发 ---")
    for i, r in enumerate(results):
        st = "PASS" if r["ok"] else "FAIL"
        print(f"    #{i+1:<2} {st:<4} setup={r['setup_ms']:.0f}ms "
              f"total={r['total_ms']:.0f}ms TTFT={r['ttft_ms']:.0f}ms "
              f"chars={r['chars']} {r['info'][:30]}")
    peak_str = ", ".join(
        f"GPU{g}: {v}MiB/{u}% (基线{baseline.get(g,0)}MiB)" for g, v, u in
        [(g, p[0], p[1]) for g, p in peak.items()])
    print(f"  峰值显存/利用率: {peak_str}")
    print(f"  {ok}/{N} PASS | wall={wall_ms:.0f}ms")

    print(f"\n=== 并发测试完成: {ok}/{N} 路成功 ===")
    return ok


async def main():
    parser = argparse.ArgumentParser(description="流式视频对话 Demo")
    parser.add_argument("--video", help="视频文件路径（回放模式）")
    parser.add_argument("--audio", default="",
                        help="人声音频 wav（替代视频音轨；视频音轨常非人声导致 TurnSense 判 invalid）")
    parser.add_argument("--realtime", action="store_true", help="实时采集模式（麦克风+摄像头）")
    parser.add_argument("--prompt", default="请描述这个视频里发生了什么，尽量详细。",
                        help="对话提示词")
    parser.add_argument("--turn-decision", choices=["model", "vad_turnsense"],
                        default="vad_turnsense",
                        help="轮次判决: model=模型自主 speak/listen, vad_turnsense=VAD+TurnSense")
    parser.add_argument("--turns", type=int, default=1, help="回放模式的对话轮数")
    parser.add_argument("--duration", type=float, default=30.0, help="实时模式时长(秒)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="视频抽帧率(帧/秒)，0=不抽帧只发音频")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最大抽帧数，0=不限(用 KV 预算保护)")
    parser.add_argument("--kv-budget", type=int, default=20000,
                        help="KV 预算(tokens)，每帧≈540，超过后停止发帧保留音频。"
                             "默认 20000 ≈ 生产每路 KV(25600) 的 78%，留余量给音频+历史")
    parser.add_argument("--max-audio-s", type=float, default=None,
                        help="限制音频提取时长(秒)，默认=完整音轨。长音频约25 tok/s，注意 KV")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="回放速度倍率，1.0=真实速度(1s音频等1s)，>1 加速发送")
    parser.add_argument("--wait-reply", type=lambda v: v.lower() in ("1","true","yes","on"),
                        default=True,
                        help="VAD+TurnSense 触发回复后是否暂停发送音频，等模型回复完成再继续(默认开)。"
                             "--wait-reply=false 关闭则持续发送不等回复")
    parser.add_argument("--vad-model", choices=["fsmn", "silero"], default="fsmn",
                        help="VAD 模型: fsmn(FunASR,默认) 或 silero")
    parser.add_argument("--vad-tail-sil", type=int, default=600,
                        help="VAD 尾静音(ms)，fsmn 的 max_end_silence_time，默认600")
    parser.add_argument("--vad-max-len", type=int, default=60000,
                        help="VAD 最大段长(ms)，fsmn 的 max_single_segment_time，默认60000")
    parser.add_argument("--vad-chunk-size", type=int, default=1000,
                        help="VAD 处理窗口(ms)，fsmn 内部窗口，默认1000(1s可流式分句)")
    parser.add_argument("--audio-chunk-ms", type=int, default=100,
                        help="demo 发送音频的 chunk 大小(ms)，默认100（VAD 需 ≥25ms，"
                             "100ms 兼顾实时性与网络开销）。回放模式即用此值")
    parser.add_argument("--ts-wait-ms", type=int, default=900,
                        help="TurnSense incomplete 等待(ms)：语义不完整时等多久，超时强制回复，默认900")
    parser.add_argument("--ts-invalid-threshold", type=float, default=0.9,
                        help="TurnSense invalid 丢弃阈值(0~1)：invalid 概率≥此值则丢弃该句不回复，默认0.9")
    parser.add_argument("--direct-backend", default="",
                        help="直连后端 WS 地址(如 ws://127.0.0.1:22500/backend)，绕过 gateway")
    parser.add_argument("--gateway", default="wss://127.0.0.1:8006/v1/realtime",
                        help="gateway WS 地址")
    parser.add_argument("--host", default="127.0.0.1",
                        help="服务主机地址（内网机器连生产用，如 192.168.89.106）")
    parser.add_argument("--port", type=int, default=0,
                        help="服务端口：22500=直连后端(ws)，8006=gateway(wss)。"
                             "默认 0=未指定，用 --direct-backend / --gateway 显式地址")
    parser.add_argument("--backend", choices=["minicpm", "qwen3omni"], default="minicpm",
                        help="后端类型：minicpm（回复以 listen 结束）或 qwen3omni（response.done 结束）")
    parser.add_argument("--system-prompt", default="你是一个友好的中文助手。",
                        help="系统提示词")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="并发路数(默认0=单路)。>0 时进入并发测试：从1递增到N，同时监控 GPU 显存/利用率")
    parser.add_argument("--gpu-ids", default="0,1",
                        help="要监控的 GPU 编号(逗号分隔)，默认 '0,1'")
    args = parser.parse_args()

    if not args.video and not args.realtime and args.concurrency <= 0:
        parser.error("需要 --video 或 --realtime 之一")

    if args.concurrency > 0 and not args.video:
        parser.error("并发测试需要 --video 提供测试音视频")

    # Qwen3-Omni 是 turn-based（VAD+TurnSense 分句触发解码），不支持模型自主
    # speak/listen（model 判决）。只有 MiniCPM 全双工才有 <|listen|> 概念。
    if args.backend == "qwen3omni" and args.turn_decision == "model":
        parser.error("qwen3omni 后端必须用 --turn-decision vad_turnsense（模型自主判决仅 MiniCPM 支持）")

    # 按 --audio-chunk-ms 重算全局发送 chunk（并发/实时模式用；回放模式直接传参）
    global CHUNK_MS, CHUNK_SAMPLES
    CHUNK_MS = max(25, args.audio_chunk_ms)
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

    # 连接目标：直连后端 或 gateway。
    # 优先级：--direct-backend / --gateway 显式地址 > --host+--port 拼接 > 默认。
    ssl_ctx = None
    direct = bool(args.direct_backend)
    if args.port > 0:
        # --host/--port 拼接生产服务地址：内网其他机器可连
        if args.port == 22500:
            # 后端直连（容器内 WS，需后端 host 可达）
            url = f"ws://{args.host}:{args.port}/backend"
            direct = True
        else:
            # gateway（HTTPS/WSS）
            url = f"wss://{args.host}:{args.port}/v1/realtime"
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

    # 并发测试模式：不建单路 client，直接并发 N 路
    if args.concurrency > 0:
        gpu_ids = tuple(int(x.strip()) for x in args.gpu_ids.split(",") if x.strip())
        await run_concurrency(
            url, ssl_ctx, direct, args.backend,
            args.turn_decision, args.system_prompt,
            vad_cfg={
                "vad_model": args.vad_model,
                "vad_tail_sil": args.vad_tail_sil,
                "vad_max_len": args.vad_max_len,
                "vad_chunk_size": args.vad_chunk_size,
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
        )
        return

    client = StreamingChatClient(url, ssl_ctx=ssl_ctx, direct=direct, backend=args.backend)
    try:
        await client.connect()
        await client.init(
            mode="full_duplex",
            system_prompt=args.system_prompt,
            turn_decision=args.turn_decision,
            config={
                "vad": {
                    "vad_model": args.vad_model,
                    "vad_tail_sil": args.vad_tail_sil,
                    "vad_max_len": args.vad_max_len,
                    "vad_chunk_size": args.vad_chunk_size,
                },
                "turnsense": {
                    "incomplete_wait_ms": args.ts_wait_ms,
                    "invalid_confidence_threshold": args.ts_invalid_threshold,
                },
            },
        )
        print(f"会话已建立: session={client.session_id[:8] if client.session_id else '?'} "
              f"url={url}")

        if args.video:
            metrics = await run_file_replay(client, args.video, args.prompt,
                                            args.turn_decision, args.turns,
                                            fps=args.fps, max_frames=args.max_frames,
                                            kv_budget_tokens=args.kv_budget,
                                            audio_path=args.audio,
                                            max_audio_s=args.max_audio_s,
                                            replay_speed=args.replay_speed,
                                            wait_reply=args.wait_reply,
                                            audio_chunk_ms=args.audio_chunk_ms)
        else:
            metrics = await run_realtime(client, args.system_prompt,
                                         args.turn_decision, args.duration,
                                         fps=args.fps, kv_budget_tokens=args.kv_budget)

        # 汇总
        if metrics:
            ttfts = [m.ttft_s for m in metrics if m.ttft_s is not None]
            speeds = [m.speed_cps for m in metrics if m.speed_cps > 0]
            print("\n=== 汇总 ===")
            if ttfts:
                print(f"  平均首字延迟: {sum(ttfts)/len(ttfts):.2f}s (min={min(ttfts):.2f}s)")
            if speeds:
                print(f"  平均生成速度: {sum(speeds)/len(speeds):.0f}ch/s")
            print(f"  对话轮数: {len(metrics)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
