"""
并发推理测试工具 — 支持文本/视频输入 + 显存追踪。

用法:
  # 纯文本并发测试（默认）
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_vram_concurrency.py

  # 视频并发测试
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_vram_concurrency.py \
      --video /data/.../video.mp4

  # 指定并发路数
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_vram_concurrency.py \
      --max-concurrency 4

  # 指定服务器地址
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_vram_concurrency.py \
      --gateway wss://192.168.89.106:8006/v1/realtime
"""
import asyncio, json, ssl, base64, subprocess, time, argparse, os
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
DEFAULT_VIDEO = "/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/assets/video/turnbased/121.mp4"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def get_vram(gpu_id=0):
    """获取指定 GPU 的显存占用 (MiB)"""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True
    )
    try:
        return int(r.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return 0


async def run_text_session(idx, gateway, timeout=60):
    """一路纯文本推理：发送 "Say hello in 3 words"，等待回复。"""
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "turn_based"}}))
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        dt_conn = (time.perf_counter() - t0) * 1000

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
                return (idx, False, dt_conn, 0, m.get("reason", "?"))

        dt_infer = (time.perf_counter() - t0) * 1000
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_conn, dt_infer, "")
    except Exception as e:
        return (idx, False, 0, 0, str(e)[:60])


async def run_video_session(idx, gateway, video_b64, timeout=300):
    """一路视频推理：发送视频 + 描述请求，等待回复。"""
    t0 = time.perf_counter()
    try:
        ws = await websockets.connect(gateway, max_size=128*1024*1024, ssl=SSL_CTX)
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({"type": "session.init",
                      "payload": {"mode": "turn_based"}}))
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        dt_conn = (time.perf_counter() - t0) * 1000

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
                return (idx, False, dt_conn, 0, m.get("reason", "?"))

        dt_infer = (time.perf_counter() - t0) * 1000
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
        return (idx, True, dt_conn, dt_infer, final_text[:80])
    except Exception as e:
        return (idx, False, 0, 0, str(e)[:60])


def format_results_table(results):
    """格式化结果表格"""
    lines = []
    lines.append("  %3s  %-6s  %8s  %8s  %s" % ("#", "status", "conn(ms)", "infer(ms)", "info"))
    lines.append("  " + "-" * 70)
    for r in results:
        idx, ok, conn, infer, info = r
        if ok:
            lines.append("  %3d  %-6s  %8.0f  %8.0f  %s" % (idx, "PASS", conn, infer, info[:40]))
        else:
            lines.append("  %3d  %-6s  %8s  %8s  %s" % (idx, "FAIL", "-", "-", info[:60]))
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="并发推理测试工具")
    parser.add_argument("--gateway", default=GATEWAY, help="WebSocket 网关地址")
    parser.add_argument("--video", nargs="?", const=DEFAULT_VIDEO, default=None,
                        help="视频文件路径（不指定则测纯文本）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="最大并发路数 (默认 4)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="每路超时秒数 (默认 300)")
    args = parser.parse_args()

    mode = "text" if args.video is None else "video"
    gateway = args.gateway

    print("=" * 70)
    print("  MiniCPM-o 并发推理测试")
    print("  - 模式: %s" % mode)
    print("  - 网关: %s" % gateway)
    print("  - GPU:  %d" % args.gpu)
    print("  - 最大并发: %d 路" % args.max_concurrency)
    if args.video:
        fsize = os.path.getsize(args.video)
        print("  - 视频: %s (%d MB)" % (args.video, fsize // 1024 // 1024))
    print("=" * 70)

    # 预载视频（如果需要）
    video_b64 = None
    if args.video:
        with open(args.video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        print("\n  视频 base64: %d MB\n" % (len(video_b64) // 1024 // 1024))

    base_vram = get_vram(args.gpu)
    print("  GPU %d 空闲显存: %d MiB\n" % (args.gpu, base_vram))

    all_ok = True
    for N in range(1, args.max_concurrency + 1):
        vram_before = get_vram(args.gpu)
        t_all = time.perf_counter()

        if mode == "text":
            tasks = [run_text_session(i, gateway, args.timeout) for i in range(N)]
        else:
            tasks = [run_video_session(i, gateway, video_b64, args.timeout) for i in range(N)]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = (time.perf_counter() - t_all) * 1000
        vram_after = get_vram(args.gpu)

        # 处理异常（asyncio.gather 的异常不会抛出，会以 Exception 形式返回）
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append((len(processed), False, 0, 0, str(r)[:60]))
            else:
                processed.append(r)

        ok = sum(1 for r in processed if r[1])
        if ok < N:
            all_ok = False

        print("  --- %d 路并发 ---" % N)
        print(format_results_table(processed))
        print("  VRAM: %d -> %d MiB (+%d MiB) | %d/%d PASS in %.0fms wall" %
              (vram_before, vram_after, vram_after - vram_before, ok, N, wall))
        print()

        if ok < N:
            print("  ⚠️  %d 路时出现失败，停止递增\n" % N)
            break

        # 等待显存释放
        await asyncio.sleep(3)

    vram_final = get_vram(args.gpu)
    max_ok = args.max_concurrency if all_ok else (ok)
    print("=" * 70)
    print("  测试完成")
    print("  最大成功并发: %d 路" % max_ok)
    print("  GPU %d 显存: %d -> %d MiB (基线 %d MiB)" %
          (args.gpu, base_vram, vram_final, base_vram))
    print("=" * 70)
    exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
