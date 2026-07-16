"""
全双工流式测试 — 模拟实时音视频输入 + 文本提问触发回复。

流程：
  1. session.init → 建立全双工会话
  2. 逐帧发送音频（模拟麦克风实时流）
  3. 发送完音频后，通过 text 字段提问，触发模型回复
  4. 收集 text/audio/listen 响应
"""
import asyncio, json, ssl, base64, os, subprocess, time
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
DATA_DIR = "/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/assets/video/fullduplex"
SR = 16000
FRAME_LEN = 1600  # 100ms
FRAME_BYTES = FRAME_LEN * 2


def extract_pcm(path):
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vn",
         "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
         "-f", "s16le", "pipe:1"], capture_output=True, check=True)
    return r.stdout


def extract_frames(path, max_n=3):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", path],
         capture_output=True, text=True)
    dur = float(probe.stdout.strip() or 30)
    n = min(max_n, max(1, int(dur / 3)))
    frames = []
    for i in range(n):
        pos = dur * (i + 1) / (n + 1)
        r = subprocess.run(["ffmpeg", "-y", "-ss", str(pos), "-i", path,
             "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1"],
             capture_output=True, check=True)
        frames.append(base64.b64encode(r.stdout).decode())
    return frames


def make_chunks(pcm):
    out = []
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i:i+FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk.ljust(FRAME_BYTES, b'\x00')
        out.append(chunk)
    return out


async def drain_response(ws, name, timeout=30):
    """接收模型响应，直到 LISTEN / response.done / session.closed。
       返回 (full_text, audio_chunks, reason)
       reason: "listen" | "done" | "text" | "closed" | "timeout"
    """
    text = ""
    audio = 0
    while True:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        except asyncio.TimeoutError:
            return text, audio, "timeout"
        t = msg.get("type")
        k = msg.get("kind", "")
            #print("[drain] type=%s kind=%s text=%s" % (t, k, msg.get("text","")[:40]))
        if t == "response.output.delta":
            if k == "text":
                text += msg.get("text", "")
            elif k == "audio":
                audio += 1
            elif k == "listen":
                return text, audio, "listen"
        elif t == "response.done":
            text = msg.get("text", "") or text
            return text, audio, "done" if not text else "text"
        elif t == "session.closed":
            return text, audio, "closed"
        elif t == "error":
            print("  [%s] error: %s" % (name, str(msg)[:200]))
            return text, audio, "closed"
    return text, audio, "timeout"


async def send_audio_frames(ws, chunks, name, video_frames=None, gap=0.15):
    """逐帧发送音频（首帧附带视频帧），每帧等待模型响应。"""
    listen_cnt = 0
    sent_bytes = 0

    for idx, chunk in enumerate(chunks):
        b64 = base64.b64encode(chunk).decode()
        sent_bytes += len(chunk)

        payload = {"type": "input.append", "input": {
            "audio": b64, "max_slice_nums": 1,
        }}
        if idx == 0 and video_frames:
            payload["input"]["video_frames"] = video_frames

        await ws.send(json.dumps(payload, ensure_ascii=False))

        # 收响应
        reason = (await drain_response(ws, name))[2]
        if reason == "listen":
            listen_cnt += 1
        elif reason == "closed":
            break

        if idx < len(chunks) - 1:
            await asyncio.sleep(gap)

    return sent_bytes, listen_cnt


async def send_text_and_reply(ws, name, text, silence_b64, timeout=60):
    """附带静音帧 + text 提问（全双工强制要求 audio 字段），收集回复。"""
    payload = {"type": "input.append", "input": {
        "audio": silence_b64,
        "text": text,
        "max_slice_nums": 1,
    }}
    await ws.send(json.dumps(payload, ensure_ascii=False))

    reply_text, reply_audio, _ = await drain_response(ws, name, timeout=timeout)
    return reply_text, reply_audio


async def run_streaming(name, chunks, video_frames=None,
                         question=None, silence_b64=None,
                         gap=0.15, max_rounds=50):
    """
    全双工流式交互：
    1. 发送 max_rounds 帧音频（首帧带视频帧）
    2. 可选：通过 text 提问触发回复
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(GATEWAY, max_size=128*1024*1024, ssl=ctx) as ws:
        await asyncio.wait_for(ws.recv(), timeout=15)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "full_duplex"}}))
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        sid = m.get("session_id", "?")
        if not sid:
            return "", 0

        t0 = time.perf_counter()

        # 阶段1：发送音频帧
        n_frames = min(len(chunks), max_rounds)
        sent_bytes, listen_cnt = await send_audio_frames(
            ws, chunks[:n_frames], name, video_frames, gap)

        # 阶段2：通过 text 提问（如果有）
        reply_text = ""
        reply_audio = 0
        if question:
            await asyncio.sleep(0.3)
            reply_text, reply_audio = await send_text_and_reply(
                ws, name, question, silence_b64)

        elapsed = (time.perf_counter() - t0) * 1000
        aud_sec = sent_bytes / (SR * 2)
        print("  [%s] audio=%.1fs(%d frames) listen=%d text_audio=%d wall=%.0fms" %
              (name, aud_sec, n_frames, listen_cnt, reply_audio, elapsed))
        return reply_text, reply_audio


async def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mp4")])
    if not files:
        print("No videos found in", DATA_DIR)
        return

    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        tag = os.path.splitext(fname)[0]

        print("\n%s" % ("=" * 60))
        print("File: %s" % fname)
        print("Extracting...")

        pcm = extract_pcm(fpath)
        chunks = make_chunks(pcm)
        frames = extract_frames(fpath)
        print("  PCM: %dKB = %d frames | keyframes: %d" %
              (len(pcm)//1024, len(chunks), len(frames)))

        # 生成一帧静音用于 text 提问时附带（全双工要求 audio 非空）
        silence_b64 = base64.b64encode(b'\x00' * FRAME_BYTES).decode()

        # 测试1：纯音频 + text 提问
        print("\n--- Test 1: 音频流 + text 提问 ---")
        text, ac = await run_streaming(tag, chunks[:20],
            video_frames=None,
            question="请问你看到了什么？",
            silence_b64=silence_b64,
            gap=0.15, max_rounds=20)
        if text:
            print("  TEXT(%d chars): %s" % (len(text), text[:200]))
        else:
            print("  (no text)")
        await asyncio.sleep(2)

        # 测试2：视频 + 音频 + text 提问
        print("\n--- Test 2: 视频帧 + 音频流 + text 提问 ---")
        text, ac = await run_streaming(tag, chunks[:20],
            video_frames=frames,
            question="请描述这个视频里发生了什么",
            silence_b64=silence_b64,
            gap=0.15, max_rounds=20)
        if text:
            print("  TEXT(%d chars): %s" % (len(text), text[:200]))
        else:
            print("  (no text)")
        print()


if __name__ == "__main__":
    asyncio.run(main())
