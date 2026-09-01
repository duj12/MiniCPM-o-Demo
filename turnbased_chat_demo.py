#!/usr/bin/env python3
"""Turn-based 音视频理解 Demo —— 一次请求发完整音视频，模型输出结构化描述。

区别于 streaming_chat_demo（full_duplex 对话），本 demo 走 /v1/realtime?mode=chat
（turn_based）：把视频帧 + 音频 + 文本 prompt 一次性发给后端，模型原生理解
画面与语音，输出视频描述（人物/行为/环境/情绪）。不依赖 VAD 语音触发。

用法：
  # 视频理解（画面 + 语音，输出结构化描述）
  python turnbased_chat_demo.py --video assets/video/turnbased/121.mp4

  # 纯音频理解（只听语音）
  python turnbased_chat_demo.py --audio xxx.wav

  # 只分析画面（不看语音）
  python turnbased_chat_demo.py --video xxx.mp4 --no-audio

  # 自定义帧数上限/音频时长
  python turnbased_chat_demo.py --video xxx.mp4 --max-frames 60 --max-audio-s 30

常用参数：
  --max-frames         视频抽帧上限，默认40。1fps 抽帧，超过上限截断
                       （每帧≈530token，40帧≈2.1万token < n_ctx 25600）
  --no-audio           只发视频帧，不发音频
  --max-audio-s        音频时长上限（秒），默认全轨
  --prompt             描述 prompt（默认内置结构化模板）
  --host / --port      gateway 地址（默认 192.168.89.106:8006）
"""

import argparse
import asyncio
import json
import ssl
import sys
from typing import List, Optional

import numpy as np

# 复用 streaming_chat_demo 的音视频提取工具（不重复实现）
from streaming_chat_demo import (
    b64,
    extract_audio_pcm,
    extract_keyframes,
    probe_duration,
)

SAMPLE_RATE = 16000

DEFAULT_PROMPT = """你是一个音视频理解助手。请综合画面与语音输出视频的结构化描述：
- 画面为主，语音辅助理解场景
- 只描述实际可见/可闻的信息，不确定的写"不可见/不确定"，不编造

【人物信息】(P0，必答)
- 性别、年龄段
- 穿着（颜色、款式）

【行为/运动】(P0，必答)
- 人物动作
- 相对距离变化：靠近/远离/静止
- 运动方向、速度

【环境】(P1)
- 场景类型（室内/室外）
- 其他人（数量、行为）
- 背景要素（物品、光线）

【情绪】(P1)
- 表情/姿态/语气反映的情绪

【语音内容】(P1)
- 说话人说的语音内容转写成文本

输出：按五类分条，每类先写关键词再写描述。"""


async def run_turnbased(url: str, ssl_ctx, video_path: str, audio_path: str,
                        prompt: str, max_frames: int,
                        use_audio: bool, max_audio_s: Optional[float]) -> None:
    """连 gateway mode=chat，发音视频 + prompt，流式打印模型描述。"""
    import websockets

    # ── 提取输入 ──
    frames: List[bytes] = []
    audio: Optional[np.ndarray] = None
    if video_path:
        # 默认 1fps 抽帧（覆盖全视频）；超过 max_frames（默认40）时限制到 max_frames，
        # 避免长视频帧数过多超出 KV 预算（每帧 ≈530 token，40 帧 ≈2.1 万 < 25600）。
        dur = probe_duration(video_path)
        n_frames = min(int(np.ceil(dur)), max_frames) if dur > 0 else max_frames
        frames = extract_keyframes(video_path, n_frames=n_frames)
        print(f"  视频帧: {len(frames)} 帧（1fps 抽帧, 视频 {dur:.0f}s, 上限 {max_frames}）")
        if use_audio:
            audio = extract_audio_pcm(video_path, max_s=max_audio_s)
    elif audio_path:
        import soundfile as sf
        if sf is None:
            print("  [warn] soundfile 不可用，无法读音频")
        else:
            a, sr = sf.read(audio_path, dtype="float32")
            audio = np.asarray(a, dtype=np.float32)
            if max_audio_s:
                audio = audio[:int(max_audio_s * sr)]
    else:
        print("  [error] 需要 --video 或 --audio")
        return

    if audio is not None:
        print(f"  音频: {len(audio)/SAMPLE_RATE:.1f}s")
    else:
        print("  音频: 未使用")

    if not frames and audio is None:
        print("  [error] 无可分析的输入")
        return

    # ── 连接 gateway mode=chat ──
    async with websockets.connect(url, ssl=ssl_ctx, max_size=256 * 1024 * 1024) as ws:
        # 等 queue_done
        while True:
            m = json.loads(await ws.recv())
            if m.get("type") in ("session.queue_done", "queue_done"):
                break
        await ws.send(json.dumps({"type": "session.init", "payload": {}}))
        while True:
            m = json.loads(await ws.recv())
            if m.get("type") == "session.created":
                break

        # ── 组装 input：对齐后端 turn_based 的 messages content 格式 ──
        # 后端 parse_one_message 认 {type:"image",data:jpegb64}（画面帧）/
        # {type:"audio",data:b64}（float32 PCM），嵌在 message 的 content 数组。
        # 顶层 video_frames/audio 是 full_duplex 格式，turn_based 不认 → 幻觉。
        # 注意：帧数不宜多（17 帧 + 音频会让后端 prefill 卡住），默认限 max_frames。
        content: List[dict] = [{"type": "text", "text": prompt}]
        for f in frames:
            content.append({"type": "image", "data": b64(f)})
        if audio is not None:
            content.append({"type": "audio", "data": b64(audio)})

        inp: dict = {
            "messages": [{"role": "user", "content": content}],
            "streaming": True,
        }

        print("\n  ── 描述 ── ", end="", flush=True)
        await ws.send(json.dumps({"type": "input.append", "input": inp}))

        # 收事件流式打印
        done = False
        while not done:
            try:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            except asyncio.TimeoutError:
                print("\n  [timeout] 等待回复超时", flush=True)
                break
            t = ev.get("type")
            if t == "response.output.delta" and ev.get("kind") == "text":
                print(ev.get("text", ""), end="", flush=True)
            elif t in ("response.done", "session.closed", "error"):
                done = True
        print()


def _ssl_ctx_noverify():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def main():
    parser = argparse.ArgumentParser(description="Turn-based 音视频理解 Demo")
    parser.add_argument("--video", default="", help="视频文件路径（抽取画面+音轨）")
    parser.add_argument("--audio", default="", help="音频文件路径（纯音频理解，或与 --video 独立用）")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="视频抽帧上限。默认1fps抽帧，超过此上限时截断（40=40s以上视频限40帧，"
                             "每帧≈530token，40帧≈2.1万token < n_ctx 25600）")
    parser.add_argument("--no-audio", action="store_true", help="只发视频帧，不发音频")
    parser.add_argument("--max-audio-s", type=float, default=None, help="音频时长上限(秒)，默认全轨")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="描述 prompt（默认内置结构化模板）")
    parser.add_argument("--host", default="192.168.89.106", help="gateway 主机")
    parser.add_argument("--port", type=int, default=8006, help="gateway 端口")
    args = parser.parse_args()

    if not args.video and not args.audio:
        parser.error("需要 --video 或 --audio")

    url = f"wss://{args.host}:{args.port}/v1/realtime?mode=chat"
    print(f"连接目标: {url}")
    asyncio.run(run_turnbased(
        url, _ssl_ctx_noverify(), args.video, args.audio,
        args.prompt, args.max_frames,
        not args.no_audio, args.max_audio_s,
    ))


if __name__ == "__main__":
    main()
