"""
并发推理测试工具 — 支持文本/视频/全双工模式 + 显存追踪。

用法:
  # 纯文本并发测试（默认）
  python tests/test_vram_concurrency.py

  # 视频并发测试 (turn_based)
  python tests/test_vram_concurrency.py --mode video

  # 全双工静音+文本（模拟麦克风）
  python tests/test_vram_concurrency.py --mode duplex

  # 全双工真实视频（提取视频音频帧+帧图片，模拟实时视频通话）
  python tests/test_vram_concurrency.py --mode duplex --video

  # 全双工真实视频 + 指定文件
  python tests/test_vram_concurrency.py --mode duplex --video /path/to/video.mp4

  # 指定并发路数和网关
  python tests/test_vram_concurrency.py --mode duplex --max-concurrency 3 \
      --gateway wss://host:8006/v1/realtime
"""
import asyncio, json, ssl, base64, subprocess, time, argparse, os
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
DEFAULT_TURN_VIDEO = "/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/assets/video/turnbased/121.mp4"
DEFAULT_DUPLEX_VIDEO = "/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/assets/video/fullduplex/121+pad.mp4"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# 全双工音频参数
SR = 16000
FRAME_MS = 100
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)  # 1600
FRAME_BYTES = FRAME_SAMPLES * 2             # 3200 bytes


def get_vram(gpu_id=0):
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True
    )
    try:
        return int(r.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return 0


# ───── 视频/音频提取 ─────

def extract_audio_frames(video_path, max_frames=0):
    """提取视频 PCM 音频并按 100ms 切帧。返回 (pcm_frames, total_seconds)"""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
         "-f", "s16le", "pipe:1"],
        capture_output=True, check=True
    )
    pcm = r.stdout
    frames = []
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i:i+FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk.ljust(FRAME_BYTES, b'\x00')
        frames.append(chunk)
    total_s = len(pcm) / (SR * 2)
    if max_frames > 0 and len(frames) > max_frames:
        frames = frames[:max_frames]
    return frames, total_s


def extract_video_frames_uniform(video_path, max_n=3):
    """从视频均匀提取 N 帧 JPEG base64（用于 turn_based 单帧输入）。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        dur = 30
    n = min(max_n, max(1, int(dur / 3)))
    frames = []
    for i in range(n):
        pos = dur * (i + 1) / (n + 1)
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(pos), "-i", video_path,
             "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1"],
            capture_output=True, check=True
        )
        frames.append(base64.b64encode(r.stdout).decode())
    return frames


def extract_video_frames_stream(video_path, fps=5):
    """模拟摄像头采集：按指定 fps 提取 JPEG 帧，返回 (frame_list, frame_index_map)。

    frame_list: 所有 JPEG base64 帧（按时间排序）
    frame_index_map: dict[audio_frame_index] = video_frame_base64
      — 告诉发送方：在第 N 个音频帧时，应该附带此视频帧。

    假设音频 100ms/帧，视频 fps 帧/秒：
      - 每 audio_frames_per_video = 10/fps 个音频帧附带一帧视频
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,r_frame_rate",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    parts = probe.stdout.strip().split(",")
    try:
        dur = float(parts[0])
    except (ValueError, IndexError):
        dur = 30

    # 按时间线均匀取帧，每秒取 fps 帧
    total_video_frames = int(dur * fps)
    frame_list = []
    for i in range(total_video_frames):
        pos = (i + 0.5) / fps  # 每帧在时间线上的中间位置
        if pos > dur:
            break
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(pos), "-i", video_path,
             "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1"],
            capture_output=True, check=True
        )
        frame_list.append(base64.b64encode(r.stdout).decode())

    # 构建映射：第 N 个音频帧 → 对应的视频帧
    audio_frames_per_second = 10  # 100ms
    video_frame_interval = audio_frames_per_second / fps  # 每几个音频帧发一帧视频
    frame_index_map = {}
    for v_idx in range(len(frame_list)):
        audio_idx = int(v_idx * video_frame_interval)
        if audio_idx not in frame_index_map:
            frame_index_map[audio_idx] = frame_list[v_idx]

    return frame_list, frame_index_map


# ───── turn_based（纯文本 / 视频） ─────

async def run_text_session(idx, gateway, timeout=60):
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "turn_based"}}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if m.get("type") == "session.created":
                break

        await ws.send(json.dumps({"type": "input.append", "input": {
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "streaming": False, "tts": {"enabled": False},
            "use_tts_template": False,
        }}, ensure_ascii=False))

        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if m.get("type") == "response.done":
                break
            elif m.get("type") in ("session.closed", "error"):
                return (idx, False, time.perf_counter() - t0, str(m.get("reason", "?")))

        dt = (time.perf_counter() - t0) * 1000
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt, dt, "")
    except Exception as e:
        return (idx, False, 0, str(e)[:60])


async def run_video_session(idx, gateway, video_b64, timeout=300):
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "turn_based"}}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if m.get("type") == "session.created":
                break

        await ws.send(json.dumps({"type": "input.append", "input": {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe this video in detail"},
                {"type": "video", "data": video_b64,
                 "name": "test.mp4", "duration": 10},
            ]}],
            "streaming": False, "tts": {"enabled": False},
            "use_tts_template": False, "omni_mode": True,
            "image": {"max_slice_nums": 1},
        }}, ensure_ascii=False))

        prev_text = ""
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if m.get("type") == "response.output.delta" and m.get("kind") == "text":
                prev_text += m.get("text", "")
            elif m.get("type") == "response.done":
                final_text = m.get("text", "") or prev_text
                break
            elif m.get("type") in ("session.closed", "error"):
                return (idx, False, time.perf_counter() - t0, m.get("reason", "?"))

        dt = (time.perf_counter() - t0) * 1000
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt, dt, final_text[:80])
    except Exception as e:
        return (idx, False, 0, str(e)[:60])


# ───── full_duplex ─────

async def run_duplex_session(idx, gateway, audio_frames, video_frame_map=None,
                              text_prompt=None, frame_gap=0.1, timeout=120):
    """
    一路全双工流式会话。
    逐帧发送音频，按 video_frame_map 的时间戳附带视频帧（模拟摄像头采集），
    接收模型响应。返回 (idx, ok, conn_ms, total_ms, info)
    """
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "full_duplex"}}))
        # 等待 session.created（可能先收到 session.queued）
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if m.get("type") == "session.created":
                break
        dt_conn = (time.perf_counter() - t0) * 1000

        text_output = ""
        audio_chunks = 0
        listen_count = 0

        for i, frame in enumerate(audio_frames):
            audio_b64 = base64.b64encode(frame).decode()
            payload = {"type": "input.append", "input": {
                "audio": audio_b64, "max_slice_nums": 1,
            }}
            if i == 0 and text_prompt:
                payload["input"]["text"] = text_prompt
            # 模拟摄像头：在该帧的时间点附带对应的视频帧
            frameb64 = video_frame_map.get(i) if video_frame_map else None
            if frameb64:
                payload["input"]["video_frames"] = [frameb64]

            await ws.send(json.dumps(payload, ensure_ascii=False))

            # 接收响应（直到 LISTEN 或 response.done）
            got_frame_done = False
            while not got_frame_done:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                except asyncio.TimeoutError:
                    return (idx, False, dt_conn, 0,
                            "timeout at frame %d/%d" % (i, len(audio_frames)))

                t = m.get("type")
                k = m.get("kind", "")

                if t == "response.output.delta":
                    if k == "text":
                        text_output += m.get("text", "")
                    elif k == "audio":
                        audio_chunks += 1
                    elif k == "listen":
                        listen_count += 1
                        got_frame_done = True
                elif t == "response.done":
                    text_output = m.get("text", "") or text_output
                    got_frame_done = True
                elif t == "session.closed":
                    return (idx, False, dt_conn, 0,
                            "closed: " + m.get("reason", ""))
                elif t == "error":
                    return (idx, False, dt_conn, 0, str(m)[:60])

            if i < len(audio_frames) - 1:
                await asyncio.sleep(frame_gap)

        dt_total = (time.perf_counter() - t0) * 1000

        # 构建信息摘要
        info_parts = []
        if text_output:
            info_parts.append(text_output[:60])
        if audio_chunks:
            info_parts.append("audio=%d" % audio_chunks)
        if not text_output and not audio_chunks:
            info_parts.append("listen=%d no-reply" % listen_count)
        info = " | ".join(info_parts)

        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_conn, dt_total, info)

    except Exception as e:
        return (idx, False, 0, 0, str(e)[:60])


# ───── 格式化 ─────

def fmt_results(results):
    lines = []
    lines.append("  %3s  %-6s  %8s  %8s  %s" % (
        "#", "status", "conn(ms)", "total(ms)", "info"))
    lines.append("  " + "-" * 80)
    for r in results:
        idx, ok, conn, total, info = r
        if ok:
            lines.append("  %3d  %-6s  %8.0f  %8.0f  %s" %
                         (idx, "PASS", conn, total, info[:60]))
        else:
            lines.append("  %3d  %-6s  %8s  %8s  %s" %
                         (idx, "FAIL", "-", "-", info[:60]))
    return "\n".join(lines)


# ───── 主流程 ─────

async def main():
    parser = argparse.ArgumentParser(description="并发推理测试工具")
    parser.add_argument("--gateway", default=GATEWAY, help="WebSocket 网关地址")
    parser.add_argument("--mode", choices=["text", "video", "duplex"],
                        default="text", help="测试模式")
    parser.add_argument("--video", nargs="?", const=True, default=None,
                        help="视频文件路径（turn_based 或 duplex 模式使用）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="最大并发路数 (默认 4)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="每路超时秒数 (默认 300)")
    parser.add_argument("--duplex-frames", type=int, default=10,
                        help="全双工模式每路音频帧数 (默认 10 = 1s, 0 = 全部)")
    parser.add_argument("--duplex-video-fps", type=float, default=5,
                        help="模拟摄像头帧率 (默认 5fps, 建议 3-10)")
    parser.add_argument("--duplex-gap", type=float, default=0.15,
                        help="全双工帧间隔秒数 (默认 0.15)")
    parser.add_argument("--duplex-text", type=str, default="请描述这个视频里发生了什么",
                        help="全双工模式首帧附带文本 (默认通用提问)")
    parser.add_argument("--stagger", type=float, default=2.0,
                        help="每路启动间隔秒数，避免并发初始化冲突 (默认 2.0)")
    args = parser.parse_args()

    gateway = args.gateway
    mode = args.mode

    print("=" * 80)
    print("  MiniCPM-o 并发推理测试")
    print("  - 模式: %s" % mode)
    print("  - 网关: %s" % gateway)
    print("  - GPU:  %d" % args.gpu)
    print("  - 最大并发: %d 路" % args.max_concurrency)
    if args.stagger > 0:
        print("  - 交错启动: %.1fs" % args.stagger)
    if mode == "duplex":
        print("  - 每路帧数: %d (%.1fs, 视频 %d fps)" % (
            args.duplex_frames, args.duplex_frames * FRAME_MS / 1000,
            args.duplex_video_fps))
        if args.video:
            video_path = args.video if isinstance(args.video, str) else DEFAULT_DUPLEX_VIDEO
            print("  - 视频源: %s" % video_path)
    print("=" * 80)

    # 预载视频
    video_b64 = None
    duplex_audio_frames = None
    duplex_video_frame_map = None

    if mode == "video":
        video_path = args.video if isinstance(args.video, str) else DEFAULT_TURN_VIDEO
        if not video_path or not os.path.exists(video_path):
            print("\n  视频文件不存在: %s" % video_path)
            return
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        print("\n  turn_based 视频: %s (%d MB base64)\n" %
              (video_path, len(video_b64) // 1024 // 1024))

    elif mode == "duplex" and args.video:
        # 全双工模式：从视频提取音频帧 + 视频关键帧
        video_path = args.video if isinstance(args.video, str) else DEFAULT_DUPLEX_VIDEO
        if not os.path.exists(video_path):
            print("\n  视频文件不存在: %s" % video_path)
            return
        print("\n  提取音视频: %s" % video_path)
        duplex_audio_frames, total_s = extract_audio_frames(
            video_path, max_frames=args.duplex_frames)
        _, duplex_video_frame_map = extract_video_frames_stream(
            video_path, fps=args.duplex_video_fps)
        print("  音频: %d 帧 (%.1fs), 视频帧映射表: %d 个触发点 (%d fps)\n" %
              (len(duplex_audio_frames), total_s,
               len(duplex_video_frame_map), args.duplex_video_fps))

    elif mode == "duplex" and not args.video:
        # 纯静音帧
        duplex_audio_frames = [b"\x00" * FRAME_BYTES for _ in range(args.duplex_frames)]
        print("\n  静音帧: %d 帧, %.1f 秒\n" %
              (len(duplex_audio_frames), len(duplex_audio_frames) * FRAME_MS / 1000))

    base_vram = get_vram(args.gpu)
    print("  GPU %d 空闲显存: %d MiB\n" % (args.gpu, base_vram))

    all_ok = True
    max_ok = 0

    for N in range(1, args.max_concurrency + 1):
        vram_before = get_vram(args.gpu)
        t_all = time.perf_counter()

        # 交错启动：每个 session 延迟 stagger 秒，避免并发初始化显存冲突
        async def start_one(idx):
            delay = idx * args.stagger
            if delay > 0:
                await asyncio.sleep(delay)
            if mode == "text":
                return await run_text_session(idx, gateway, args.timeout)
            elif mode == "video":
                return await run_video_session(idx, gateway, video_b64, args.timeout)
            else:  # duplex
                text_prompt = args.duplex_text if not args.video \
                              else "请描述这个视频里发生了什么"
                return await run_duplex_session(
                    idx, gateway, duplex_audio_frames,
                    video_frame_map=duplex_video_frame_map,
                    text_prompt=text_prompt,
                    frame_gap=args.duplex_gap,
                    timeout=args.timeout)

        tasks = [start_one(i) for i in range(N)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        wall = (time.perf_counter() - t_all) * 1000
        vram_after = get_vram(args.gpu)

        processed = []
        for r in raw:
            if isinstance(r, Exception):
                processed.append((len(processed), False, 0, 0, str(r)[:60]))
            else:
                processed.append(r)

        ok = sum(1 for r in processed if r[1])
        if ok == N:
            max_ok = N
        else:
            all_ok = False

        print("  --- %d 路并发 ---" % N)
        print(fmt_results(processed))
        print("  VRAM: %d -> %d MiB (+%d MiB) | %d/%d PASS in %.0fms wall" %
              (vram_before, vram_after, vram_after - vram_before, ok, N, wall))
        print()

        if ok < N:
            print("  ⚠️  %d 路时出现失败，停止递增\n" % N)
            break

        await asyncio.sleep(4)

    vram_final = get_vram(args.gpu)
    print("=" * 80)
    print("  测试完成")
    print("  最大成功并发: %d 路" % max_ok)
    print("  GPU %d 显存: %d -> %d MiB (基线 %d MiB)" %
          (args.gpu, base_vram, vram_final, base_vram))
    print("=" * 80)
    # 短暂等待，避免退出时 SSL 未完全释放导致 Fatal error on SSL transport
    await asyncio.sleep(0.5)
    exit(0 if all_ok else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
