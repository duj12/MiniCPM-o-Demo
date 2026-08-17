#!/usr/bin/env python3
"""流式视频对话 Demo — 支持音视频文件回放 / 实时采集，两种 turn_decision 判决。

复用 MiniCPM-o-Demo 的全双工流式协议（/v1/realtime?mode=video）：
  - 音视频持续逐帧发送（"边听边看"，累积进 KV cache）
  - turn_decision="model"        : 模型自主决定 speak/listen（现有全双工）
  - turn_decision="vad_turnsense": VAD 检测语音停顿 + TurnSense 语义完整后才回复
    （worker 端用 force_listen 累积，说完才触发 decode —— 首字延迟更低）

用法：
  # 文件回放（视频理解，VAD+TurnSense 判决）
  python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
      --turn-decision vad_turnsense --prompt "描述这个视频"

  # 实时采集（麦克风 + 摄像头，模型自主判决）
  python streaming_chat_demo.py --realtime --turn-decision model

  # 直连后端（绕过 gateway，调试用）
  python streaming_chat_demo.py --video xxx.mp4 --direct-backend ws://127.0.0.1:22500/backend

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
CHUNK_MS = 1000                                       # 视频帧/音频块节奏
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
                 direct: bool = False):
        self.url = url
        self.ssl_ctx = ssl_ctx
        self.max_size = max_size
        self.direct = direct
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
                        # 实时打印模型回复文本（流式）
                        print(f"  [模型] {chunk}", end="", flush=True)
                elif kind == "listen":
                    m.model_state = "listening"
                    # 模型选择聆听（无回复）——记录时间，若持续过久则放弃等待
                    if first_delta_at is None:
                        last_listen_at = time.monotonic()
                        print("  [模型] Listen（选择聆听，未回复）", flush=True)
                    else:
                        last_listen_at = None  # 已开始回复，listen 是回复结束标志
                elif kind == "audio":
                    m.model_state = "speaking"
            elif t == "response.done":
                text = ev.get("text", "") or text
                done = True
            elif t == "session.closed":
                break
            elif t == "session.error":
                print(f"  [error] {ev}")
                break

        m.reply_chars = len(text)
        m.reply_s = (time.monotonic() - t_start) if done else (time.monotonic() - t_start)
        if reply_started and m.reply_s > 0:
            m.speed_cps = m.reply_chars / m.reply_s
        elif m.reply_chars > 0:
            m.speed_cps = 0.0
        m.reply_text = text
        return m

    async def close(self, reason: str = "demo_done"):
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
                          kv_budget_tokens: int = 6000,
                          audio_path: str = "",
                          max_audio_s: Optional[float] = None,
                          replay_speed: float = 1.0):
    """离线：按 fps 流式抽帧 + 音频切块发送，模拟实时流。

    帧策略：按 fps 持续抽帧（max_frames=0 不限制总数），但用 KV 预算保护
    ——每帧 ≈ 540 token（缩放后），累计超过 kv_budget_tokens 后停止发帧
    （音频继续），避免撑爆 KV cache。这样"最新画面"始终在流，旧帧滚动丢弃。

    audio_path: 指定人声音频 wav（替代视频音轨）。视频音轨常为非人声背景音，
      TurnSense 会判 invalid；用清晰人声可验证 VAD+TurnSense 完整链路。
    max_audio_s: 限制音频提取时长（秒）。None=完整音轨。注意长音频 token
      量巨大（约 25 tok/s），可能超出 KV，需配合 --kv-budget / 后端 -c。
    """
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
    all_metrics = []
    for turn in range(n_turns):
        m = TurnMetrics(turn_idx=turn + 1, in_audio_s=audio_s, in_video_s=video_s)
        sent_frames = 0

        # ── 边发边收：逐块发送音频+帧，同时检查 worker 是否已触发回复 ──
        # vad_turnsense 下回复时机由 worker 的 VAD+TurnSense 决定（自动触发）。
        # model 下后端自主 speak/listen，无需手动触发帧。
        n_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMPLES)))
        reply_done = False
        rm = TurnMetrics(turn_idx=turn + 1)
        for i in range(n_chunks + 1):
            if i < n_chunks:
                chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                frame_b64 = []
                if sent_frames * TOKENS_PER_FRAME < kv_budget_tokens and frames:
                    frame_b64 = [b64(frames[i % len(frames)])]
                    sent_frames += 1

                text = prompt if i == 0 else ""
                await client.send_input(
                    audio_b64=b64(chunk), video_frames=frame_b64, text=text,
                    force_listen=(i > 0),
                )

            # 检查是否已有回复（非阻塞，短超时）
            try:
                rm = await asyncio.wait_for(
                    client.collect_reply(timeout=0.4, listen_giveup_s=0.4),
                    timeout=0.5,
                )
                if rm.reply_chars > 0 or rm.model_state == "speaking":
                    reply_done = True
                    break
            except (asyncio.TimeoutError, TimeoutError):
                pass

            if i < n_chunks:
                # 按真实音频时长节奏发送（发 1s 音频 ≈ 等 1s/replay_speed），
                # 让 worker 的 VAD 有真实处理节奏。replay_speed>1 加速（更快触发）。
                await asyncio.sleep(CHUNK_MS / 1000.0 / replay_speed)

        m.in_video_s = min(video_s, sent_frames / max(fps, 0.1))
        if not reply_done:
            # 发送完毕仍无回复：
            #   vad_turnsense: 补发一段静音让 worker 的 VAD 结束当前语音段
            #     （VAD 需静音持续 >min_silence_duration_ms 才触发 TurnSense → 回复）。
            #   model: 发静音触发帧让后端 decode。
            if turn_decision == "vad_turnsense":
                for _ in range(2):  # 2s 静音，确保超过 VAD 静音阈值
                    await client.send_input(audio_b64=b64(np.zeros(CHUNK_SAMPLES, dtype=np.float32)),
                                            force_listen=True)
                    await asyncio.sleep(0.1)
            else:
                await client.send_input(audio_b64=b64(np.zeros(CHUNK_SAMPLES, dtype=np.float32)))
            rm = await client.collect_reply(listen_giveup_s=15.0)

        m.ttft_s = rm.ttft_s
        m.reply_s = rm.reply_s
        m.reply_chars = rm.reply_chars
        m.speed_cps = rm.speed_cps
        m.model_state = rm.model_state
        m.reply_text = rm.reply_text
        all_metrics.append(m)
        print(f"  {m.summary}")
        if m.reply_text:
            print(f"  ── 完整回复 ──\n  {m.reply_text}\n  ──────────────")

    return all_metrics


# ============================================================================
# 实时采集模式
# ============================================================================

async def run_realtime(client: StreamingChatClient, system_prompt: str,
                       turn_decision: str, duration_s: float = 30.0,
                       fps: float = 1.0, kv_budget_tokens: int = 6000):
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
                                        force_listen=True)
                cur_audio += 1.0

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
    parser.add_argument("--kv-budget", type=int, default=6000,
                        help="KV 预算(tokens)，每帧≈540，超过后停止发帧保留音频")
    parser.add_argument("--max-audio-s", type=float, default=None,
                        help="限制音频提取时长(秒)，默认=完整音轨。长音频约25 tok/s，注意 KV")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="回放速度倍率，1.0=真实速度(1s音频等1s)，>1 加速发送")
    parser.add_argument("--direct-backend", default="",
                        help="直连后端 WS 地址(如 ws://127.0.0.1:22500/backend)，绕过 gateway")
    parser.add_argument("--gateway", default="wss://127.0.0.1:8006/v1/realtime",
                        help="gateway WS 地址")
    parser.add_argument("--system-prompt", default="你是一个友好的中文助手。",
                        help="系统提示词")
    args = parser.parse_args()

    if not args.video and not args.realtime:
        parser.error("需要 --video 或 --realtime 之一")

    # 连接目标：直连后端 或 gateway
    ssl_ctx = None
    direct = bool(args.direct_backend)
    url = args.direct_backend or args.gateway
    if not direct:
        # gateway 是 HTTPS/WSS，需要 no-verify + mode 参数
        ssl_ctx = _ssl_ctx_noverify()
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}mode=video"
    elif args.direct_backend.startswith("wss"):
        ssl_ctx = _ssl_ctx_noverify()

    client = StreamingChatClient(url, ssl_ctx=ssl_ctx, direct=direct)
    try:
        await client.connect()
        await client.init(
            mode="full_duplex",
            system_prompt=args.system_prompt,
            turn_decision=args.turn_decision,
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
                                            replay_speed=args.replay_speed)
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
