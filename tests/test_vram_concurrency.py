"""
并发推理测试工具 — 支持文本/视频/全双工模式 + 显存追踪 + 延迟/吞吐指标。

用法:
  # 纯文本并发测试（默认）
  python tests/test_vram_concurrency.py

  # 视频并发测试 (turn_based)
  python tests/test_vram_concurrency.py --mode video

  # 全双工静音+文本（模拟麦克风）
  python tests/test_vram_concurrency.py --mode duplex

  # 全双工真实视频（提取视频音频帧+帧图片，模拟实时视频通话）
  python tests/test_vram_concurrency.py --mode duplex --video

  # 指定并发路数和网关
  python tests/test_vram_concurrency.py --mode duplex --max-concurrency 3 \
      --gateway wss://host:8006/v1/realtime
"""
import asyncio, json, ssl, base64, subprocess, time, argparse, os
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
DEFAULT_TURN_VIDEO = "assets/video/turnbased/121.mp4"
DEFAULT_DUPLEX_VIDEO = "assets/video/turnbased/121.mp4"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

SR = 16000
FRAME_MS = 1000
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def get_vram(gpu_id=0):
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return 0


def get_gpu_stats(gpu_id=0):
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True)
    try:
        parts = r.stdout.strip().split(", ")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


# ───── 视频/音频提取 ─────

def extract_audio_frames(video_path, max_frames=0):
    r = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
        "-f", "s16le", "pipe:1"], capture_output=True, check=True)
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


def extract_video_frame(video_path, offset=0.5):
    """Extract 1 JPEG frame at time offset [0-1], matching frontend."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True)
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        dur = 30
    pos = dur * offset
    r = subprocess.run(["ffmpeg", "-y", "-ss", str(pos), "-i", video_path,
        "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1"],
        capture_output=True, check=True)
    return base64.b64encode(r.stdout).decode(), dur


def extract_video_frames_stream(video_path, fps=5):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True)
    parts = probe.stdout.strip().split(",")
    try:
        dur = float(parts[0])
    except (ValueError, IndexError):
        dur = 30
    total_video_frames = int(dur * fps)
    frame_list = []
    for i in range(total_video_frames):
        pos = (i + 0.5) / fps
        if pos > dur:
            break
        r = subprocess.run(["ffmpeg", "-y", "-ss", str(pos), "-i", video_path,
            "-vframes", "1", "-q:v", "5", "-f", "mjpeg", "pipe:1"],
            capture_output=True, check=True)
        frame_list.append(base64.b64encode(r.stdout).decode())

    audio_frames_per_second = 10
    video_frame_interval = audio_frames_per_second / fps
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
        dt_setup = (time.perf_counter() - t0) * 1000

        await ws.send(json.dumps({"type": "input.append", "input": {
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "streaming": True, "tts": {"enabled": False},
            "use_tts_template": False,
        }}, ensure_ascii=False))
        t_send = time.perf_counter()

        text = ""
        text_done_ts = None
        first_token_ts = None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            t = m.get("type")
            if t == "response.output.delta" and m.get("kind") == "text":
                chunk = m.get("text", "")
                if chunk and first_token_ts is None:
                    first_token_ts = time.perf_counter()
                text += chunk
            elif t == "response.done":
                final_text = m.get("text", "") or text
                if final_text and first_token_ts is None:
                    first_token_ts = time.perf_counter()
                text = final_text
                text_done_ts = time.perf_counter()
                break
            elif t in ("session.closed", "error"):
                return (idx, False, dt_setup, 0, 0, 0, m.get("reason", "?"))

        total_ms = (text_done_ts - t_send) * 1000 if text_done_ts else 0
        ft_ms = (first_token_ts - t_send) * 1000 if first_token_ts else 0
        gen_ms = total_ms  # non-streaming: all text arrives at once
        cps = len(text) / (gen_ms / 1000.0) if gen_ms > 50 else 0
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_setup, ft_ms, gen_ms, len(text),
                "%dchars %.0fch/s" % (len(text), cps) if text else "")
    except Exception as e:
        return (idx, False, 0, 0, 0, 0, str(e)[:60])


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
        dt_setup = (time.perf_counter() - t0) * 1000

        await ws.send(json.dumps({"type": "input.append", "input": {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe this video in detail"},
                {"type": "video", "data": video_b64,
                 "name": "test.mp4", "duration": 10},
            ]}],
            "streaming": True, "tts": {"enabled": False},
            "use_tts_template": False, "omni_mode": True,
            "image": {"max_slice_nums": 1},
        }}, ensure_ascii=False))
        t_send = time.perf_counter()

        text = ""
        text_done_ts = None
        first_token_ts = None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            t = m.get("type")
            if t == "response.output.delta" and m.get("kind") == "text":
                chunk = m.get("text", "")
                if chunk and first_token_ts is None:
                    first_token_ts = time.perf_counter()
                text += chunk
            elif t == "response.done":
                final_text = m.get("text", "") or text
                if final_text and first_token_ts is None:
                    first_token_ts = time.perf_counter()
                text = final_text
                text_done_ts = time.perf_counter()
                break
            elif t in ("session.closed", "error"):
                return (idx, False, dt_setup, 0, 0, 0, m.get("reason", "?"))

        total_ms = (text_done_ts - t_send) * 1000 if text_done_ts else 0
        ft_ms = (first_token_ts - t_send) * 1000 if first_token_ts else 0
        gen_ms = total_ms  # non-streaming: all text arrives at once
        cps = len(text) / (gen_ms / 1000.0) if gen_ms > 50 else 0
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_setup, ft_ms, gen_ms, len(text),
                "%dchars %.0fch/s" % (len(text), cps) if text else "")
    except Exception as e:
        return (idx, False, 0, 0, 0, 0, str(e)[:60])


# ───── full_duplex ─────

async def run_duplex_session(idx, gateway, audio_frames, video_frame_map=None,
                              text_prompt=None, frame_gap=0.1, timeout=120):
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {
                          "mode": "full_duplex",
                          "system_prompt": "Streaming Omni Conversation.",
                          "config": {"length_penalty": 1.1},
                          "max_slice_nums": 1,
                          "use_tts": True,
                      }}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if m.get("type") == "session.created":
                break
        dt_setup = (time.perf_counter() - t0) * 1000

        text_output = ""
        audio_chunks = 0
        listen_count = 0
        first_text_ts = None
        first_audio_ts = None
        t_send = time.perf_counter()

        for i, frame in enumerate(audio_frames):
            audio_b64 = base64.b64encode(frame).decode()
            payload = {"type": "input.append", "input": {
                "audio": audio_b64, "max_slice_nums": 1,
            }}
            if i == 0 and text_prompt:
                payload["input"]["text"] = text_prompt
            frameb64 = video_frame_map.get(i) if video_frame_map else None
            if frameb64:
                payload["input"]["video_frames"] = [frameb64]

            await ws.send(json.dumps(payload, ensure_ascii=False))

            got_frame_done = False
            while not got_frame_done:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                except asyncio.TimeoutError:
                    return (idx, False, dt_setup, 0,
                            "timeout at frame %d/%d" % (i, len(audio_frames)))

                t = m.get("type")
                k = m.get("kind", "")

                if t == "response.output.delta":
                    now = time.perf_counter()
                    if k == "text":
                        chunk = m.get("text", "")
                        if chunk and first_text_ts is None:
                            first_text_ts = now
                        text_output += chunk
                    elif k == "audio":
                        audio_chunks += 1
                        if first_audio_ts is None:
                            first_audio_ts = now
                    elif k == "listen":
                        listen_count += 1
                        got_frame_done = True
                elif t == "response.done":
                    text_output = m.get("text", "") or text_output
                    got_frame_done = True
                elif t == "session.closed":
                    return (idx, False, dt_setup, 0, 0, 0,
                            "closed: " + m.get("reason", ""))
                elif t == "error":
                    return (idx, False, dt_setup, 0, 0, 0, str(m)[:60])

            if i < len(audio_frames) - 1:
                await asyncio.sleep(frame_gap)

        # Keep sending silence+last frame after video ends, until model finishes
        last_frame_b64 = None
        if video_frame_map and 0 in video_frame_map:
            last_frame_b64 = video_frame_map[0]
        tail_idx = 0
        model_spoke = False
        while not model_spoke and tail_idx < 100:
            b64 = base64.b64encode(b'\x00' * FRAME_BYTES).decode()
            payload = {"type": "input.append", "input": {
                "audio": b64, "max_slice_nums": 1,
            }}
            if last_frame_b64:
                payload["input"]["video_frames"] = [last_frame_b64]
                last_frame_b64 = None
            await ws.send(json.dumps(payload, ensure_ascii=False))
            tail_idx += 1

            done_reading = False
            while not done_reading:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                except asyncio.TimeoutError:
                    done_reading = True
                    break
                t = m.get("type"); k = m.get("kind","")
                if t == "response.output.delta":
                    now = time.perf_counter()
                    if k == "text":
                        chunk = m.get("text","")
                        if chunk and first_text_ts is None: first_text_ts = now
                        text_output += chunk
                        model_spoke = True
                    elif k == "audio":
                        audio_chunks += 1
                        if first_audio_ts is None: first_audio_ts = now
                        model_spoke = True
                    elif k == "listen":
                        done_reading = True
                elif t == "response.done":
                    text_output = m.get("text","") or text_output
                    if not first_text_ts and text_output: first_text_ts = time.perf_counter()
                    done_reading = True
                    break
                elif t in ("session.closed","error"):
                    done_reading = True
                    break
            await asyncio.sleep(frame_gap)

        # Once model speaks, wait for response.done
        if model_spoke:
            try:
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                    t = m.get("type")
                    if t == "response.done":
                        text_output = m.get("text","") or text_output
                        break
                    elif t in ("session.closed","error"): break
            except asyncio.TimeoutError:
                pass

        dt_total = (time.perf_counter() - t0) * 1000
        ttft_ms = (first_text_ts - t_send) * 1000 if first_text_ts else 0
        aud_first_ms = (first_audio_ts - t_send) * 1000 if first_audio_ts else 0

        info_parts = []
        if text_output:
            info_parts.append("text=%dch" % len(text_output))
            if len(text_output) <= 60:
                info_parts.append("msg=%s" % text_output)
            else:
                info_parts.append("msg=%s..." % text_output[:60])
        if audio_chunks:
            info_parts.append("audio=%d" % audio_chunks)
        if ttft_ms:
            info_parts.append("TTFT=%.0fms" % ttft_ms)
        if aud_first_ms:
            info_parts.append("AUDIO=%.0fms" % aud_first_ms)
        if not text_output and not audio_chunks:
            info_parts.append("listen=%d no-reply" % listen_count)
        info = " | ".join(info_parts)

        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_setup, dt_total, ttft_ms, aud_first_ms, info)

    except Exception as e:
        return (idx, False, 0, 0, 0, 0, str(e)[:60])


# ───── 格式化 ─────

def fmt_results(results, peak_vram=None, peak_util=None, is_turn_based=False):
    extra = ""
    if peak_vram is not None:
        extra = "  (peak VRAM: %d MiB | GPU util: %d%%)" % (peak_vram, peak_util)

    if is_turn_based:
        lines = []
        lines.append("  %3s  %-6s  %8s  %8s  %8s  %8s  %s" % (
            "#", "status", "setup(ms)", "TTFT(ms)", "gen(ms)", "chars", "ch/s"))
        lines.append("  " + "-" * 95)
        for r in results:
            idx, ok, conn, ft_ms, gen_ms, nchars, info = r
            if ok and ft_ms > 0:
                cps = nchars / (gen_ms / 1000.0) if gen_ms > 0 else 0
                lines.append("  %3d  %-6s  %8.0f  %8.0f  %8.0f  %8d  %6.0f" %
                             (idx, "PASS", conn, ft_ms, gen_ms, nchars, cps))
            elif ok:
                lines.append("  %3d  %-6s  %8.0f  %8s  %8s  %8d  %6s" %
                             (idx, "PASS", conn, "-", "-", nchars, "-"))
            else:
                lines.append("  %3d  %-6s  %8s  %8s  %8s  %8s  %8s  %s" %
                             (idx, "FAIL", "-", "-", "-", "-", "-", info[:30]))
    else:
        lines = []
        lines.append("  %3s  %-6s  %8s  %8s  %8s  %s" % (
            "#", "status", "setup(ms)", "total(ms)", "TTFT(ms)", "info"))
        lines.append("  " + "-" * 95)
        for r in results:
            idx, ok, conn, total, ttft, aud, info = r
            if ok:
                ttft_str = ("%.0f" % ttft) if ttft > 0 else "-"
                lines.append("  %3d  %-6s  %8.0f  %8.0f  %8s  %s" %
                             (idx, "PASS", conn, total, ttft_str, info[:60]))
            else:
                lines.append("  %3d  %-6s  %8s  %8s  %8s  %s" %
                             (idx, "FAIL", "-", "-", "-", info[:60]))

    lines.append(extra)
    return "\n".join(lines)


# ───── 主流程 ─────

async def main():
    parser = argparse.ArgumentParser(description="并发推理测试工具")
    parser.add_argument("--gateway", default=GATEWAY, help="WebSocket 网关地址")
    parser.add_argument("--mode", choices=["text", "video", "duplex"],
                        default="text", help="测试模式")
    parser.add_argument("--video", nargs="?", const=True, default=None,
                        help="视频文件路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="最大并发路数 (默认 4)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="每路超时秒数 (默认 300)")
    parser.add_argument("--duplex-frames", type=int, default=10,
                        help="全双工模式每路音频帧数 (默认 10 = 10s, 0 = 全部)")
    parser.add_argument("--duplex-video-fps", type=float, default=5,
                        help="模拟摄像头帧率 (默认 5fps)")
    parser.add_argument("--duplex-gap", type=float, default=0.9,
                        help="全双工帧间隔秒数 (默认 0.15)")
    parser.add_argument("--duplex-text", type=str, default="",
                        help="全双工模式首帧附带文本")
    parser.add_argument("--stagger", type=float, default=2.0,
                        help="每路启动间隔秒数 (默认 2.0)")
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
        print("  - 每路帧数: %d (%.1fs)" % (
            args.duplex_frames, args.duplex_frames * FRAME_MS / 1000))
    print("=" * 80)

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
        print("\n  视频: %s (%d MB base64)\n" %
              (video_path, len(video_b64) // 1024 // 1024))

    elif mode == "duplex" and args.video:
        video_path = args.video if isinstance(args.video, str) else DEFAULT_DUPLEX_VIDEO
        if not os.path.exists(video_path):
            print("\n  视频文件不存在: %s" % video_path)
            return
        print("\n  提取音视频: %s" % video_path)
        duplex_audio_frames, total_s = extract_audio_frames(
            video_path, max_frames=args.duplex_frames)
        frame_b64, video_dur = extract_video_frame(video_path, offset=0.5)
        # Frame map: only first audio chunk carries video frame (matches frontend)
        duplex_video_frame_map = {0: frame_b64}
        # padAfter: frontend default 2s silence after video audio
        pad_after = 2
        for _ in range(pad_after):
            duplex_audio_frames.append(b'\x00' * FRAME_BYTES)
        print("  音频: %d 帧 (%.1fs视频 + %ds静音) | 视频帧: 1 张\n" %
              (len(duplex_audio_frames), total_s, pad_after))

    elif mode == "duplex" and not args.video:
        duplex_audio_frames = [b"\x00" * FRAME_BYTES for _ in range(args.duplex_frames)]
        print("\n  静音帧: %d 帧, %.1f 秒\n" %
              (len(duplex_audio_frames), len(duplex_audio_frames) * FRAME_MS / 1000))

    base_vram = get_vram(args.gpu)
    print("  GPU %d 空闲显存: %d MiB\n" % (args.gpu, base_vram))

    all_ok = True
    max_ok = 0
    is_turn_based = mode in ("text", "video")

    for N in range(1, args.max_concurrency + 1):
        vram_before = get_vram(args.gpu)
        t_all = time.perf_counter()

        async def start_one(idx):
            delay = idx * args.stagger
            if delay > 0:
                await asyncio.sleep(delay)
            if mode == "text":
                return await run_text_session(idx, gateway, args.timeout)
            elif mode == "video":
                return await run_video_session(idx, gateway, video_b64, args.timeout)
            else:
                text_prompt = args.duplex_text
                return await run_duplex_session(
                    idx, gateway, duplex_audio_frames,
                    video_frame_map=duplex_video_frame_map,
                    text_prompt=text_prompt,
                    frame_gap=args.duplex_gap,
                    timeout=args.timeout + 120)

        peak_vram = 0
        peak_util = 0
        monitor_running = True

        async def gpu_monitor():
            nonlocal peak_vram, peak_util
            while monitor_running:
                v, u = get_gpu_stats(args.gpu)
                if v > peak_vram: peak_vram = v
                if u > peak_util: peak_util = u
                await asyncio.sleep(0.3)

        mon_task = asyncio.create_task(gpu_monitor())
        tasks = [start_one(i) for i in range(N)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        monitor_running = False
        await mon_task

        wall = (time.perf_counter() - t_all) * 1000
        vram_after = get_vram(args.gpu)

        processed = []
        for r in raw:
            if isinstance(r, Exception):
                if is_turn_based:
                    processed.append((len(processed), False, 0, 0, 0, 0, str(r)[:60]))
                else:
                    processed.append((len(processed), False, 0, 0, 0, 0, str(r)[:60]))
            else:
                processed.append(r)

        ok = sum(1 for r in processed if r[1])
        if ok == N:
            max_ok = N
        else:
            all_ok = False

        print("  --- %d 路并发 ---" % N)
        print(fmt_results(processed, peak_vram, peak_util, is_turn_based))
        print("  VRAM baseline: %d MiB | peak VRAM: %d MiB | GPU util: %d%% | %d/%d PASS in %.0fms wall" %
              (vram_before, peak_vram, peak_util, ok, N, wall))
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
    await asyncio.sleep(0.5)
    exit(0 if all_ok else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
