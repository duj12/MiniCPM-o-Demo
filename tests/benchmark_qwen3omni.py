#!/usr/bin/env python3
"""Qwen3-Omni C++ 后端性能测试

直接连接 llama-qwen3omni-server 的 WebSocket /backend 端口，
测量 TTFT、生成速度、显存占用、GPU 利用率。

用法：
    ssh 192.168.89.106
    /home/dujing/miniconda3/envs/py310/bin/python benchmark_qwen3omni.py

需要环境变量：
    BACKEND_URL=ws://127.0.0.1:22500/backend  (默认)
"""

import asyncio, json, time, os, sys, base64, subprocess, argparse
from datetime import datetime
from pathlib import Path

BACKEND_URL = os.environ.get("BACKEND_URL", "ws://127.0.0.1:22500/backend")


def gpu_info() -> dict:
    """通过 nvidia-smi 采集 GPU 状态"""
    info = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 7:
                info["gpu_id"] = int(parts[0])
                info["memory_used_mib"] = float(parts[1])
                info["memory_total_mib"] = float(parts[2])
                info["gpu_util_pct"] = float(parts[3])
                info["mem_util_pct"] = float(parts[4])
                info["temp_c"] = int(parts[5])
                info["power_w"] = float(parts[6])
    except Exception:
        pass
    return info


async def benchmark_turn(text: str, max_tokens: int = 128) -> dict:
    """每轮新开 WS 连接，执行一次 turn-based 推理"""
    import websockets
    async with websockets.connect(BACKEND_URL, max_size=128*1024*1024) as ws:
        init = {"type": "session.init", "payload": {"mode": "turn_based", "use_tts": False}}
        await ws.send(json.dumps(init))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "session.created"

        append = {
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": text}],
                "streaming": True,
                "generation": {"max_new_tokens": max_tokens},
            },
        }
        t0 = time.time()
        await ws.send(json.dumps(append))

        full_text = ""
        first_token_time = None
        while True:
            msg = json.loads(await ws.recv())
            t = msg.get("type")
            if t == "response.output.delta":
                chunk = msg.get("text", "")
                if first_token_time is None and chunk:
                    first_token_time = time.time() - t0
                full_text += chunk
            elif t == "response.done":
                total_time = time.time() - t0
                if first_token_time is None:
                    first_token_time = total_time
                break
            elif t in ("session.error", "session.closed"):
                break

    char_count = len(full_text)
    gen_time = total_time - first_token_time if first_token_time else total_time
    speed = char_count / gen_time if gen_time > 0 else 0
    return {
        "ttft_s": round(first_token_time, 3),
        "total_s": round(total_time, 3),
        "gen_s": round(gen_time, 3),
        "chars": char_count,
        "speed_cps": round(speed, 1),
        "text": full_text[:100],
    }


async def benchmark_loop(n: int, text: str, max_tokens: int) -> list:
    results = []
    for i in range(n):
        r = await benchmark_turn(text, max_tokens)
        results.append(r)
        print(f"  [{i+1}/{n}] TTFT={r['ttft_s']:.2f}s  gen={r['gen_s']:.2f}s  "
              f"{r['speed_cps']:.0f}ch/s  total={r['total_s']:.2f}s")
    return results


def print_results(label: str, results: list):
    if not results:
        return
    ttfts = [r["ttft_s"] for r in results]
    totals = [r["total_s"] for r in results]
    speeds = [r["speed_cps"] for r in results]
    gens = [r["gen_s"] for r in results]
    print(f"\n=== {label} ===")
    print(f"  TTFT:        mean={sum(ttfts)/len(ttfts):.2f}s  min={min(ttfts):.2f}s  max={max(ttfts):.2f}s")
    print(f"  生成耗时:    mean={sum(gens)/len(gens):.2f}s")
    print(f"  总耗时:      mean={sum(totals)/len(totals):.2f}s")
    print(f"  生成速度:    mean={sum(speeds)/len(speeds):.0f}ch/s")
    print(f"  回复长度:    mean={sum(r['chars'] for r in results)/len(results):.0f}ch  "
          f"({sum(r['chars'] for r in results)} total)")


def video_benchmark_text() -> str:
    return "Describe what's happening in this video in detail. What objects, people, and actions do you see?"


async def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni C++ backend benchmark")
    parser.add_argument("--runs", type=int, default=3, help="每轮测试重复次数")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成 token 数")
    parser.add_argument("--text", default="Write a short story about a robot learning to paint (50 words)")
    parser.add_argument("--video", help="视频文件路径 (测试视频理解)")
    parser.add_argument("--image", help="图片文件路径 (测试图片理解)")
    args = parser.parse_args()

    import websockets

    print("=" * 60)
    print(f"Qwen3-Omni C++ Backend Benchmark")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"Backend: {BACKEND_URL}")
    print(f"重复次数: {args.runs}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)

    # 采集基线 GPU 信息
    gpu0 = gpu_info()
    if gpu0:
        print(f"\nGPU 基线: {gpu0.get('memory_used_mib', 0):.0f} MiB / {gpu0.get('memory_total_mib', 0):.0f} MiB  "
              f"temp={gpu0.get('temp_c', 'N/A')}°C")

    # ===== 1. 纯文本 =====
    print(f"\n# 测试 1: 纯文本 ({args.text[:40]}...)")
    results = await benchmark_loop(args.runs, args.text, args.max_tokens)
    print_results("纯文本", results)
    gpu1 = gpu_info()
    if gpu1:
        print(f"  GPU 显存: {gpu1.get('memory_used_mib', 0):.0f} MiB  "
              f"利用率: {gpu1.get('gpu_util_pct', 'N/A')}%")

    # ===== 2. 图片理解 =====
    if args.image and os.path.exists(args.image):
        print(f"\n# 测试 2: 图片理解 ({args.image})")
        with open(args.image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        async with websockets.connect(BACKEND_URL, max_size=128*1024*1024) as ws:
            # session.init
            init = {"type": "session.init", "payload": {"mode": "turn_based", "use_tts": False}}
            await ws.send(json.dumps(init))
            await ws.recv()

            append = {
                "type": "input.append",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "data": img_b64},
                            {"type": "text", "text": "请详细描述这张图片"},
                        ],
                    }],
                    "streaming": True,
                    "generation": {"max_new_tokens": args.max_tokens},
                },
            }
            t0 = time.time()
            await ws.send(json.dumps(append))
            full_text = ""
            first_ts = None
            while True:
                msg = json.loads(await ws.recv())
                t = msg.get("type")
                if t == "response.output.delta":
                    chunk = msg.get("text", "")
                    if first_ts is None and chunk:
                        first_ts = time.time() - t0
                    full_text += chunk
                elif t == "response.done":
                    total = time.time() - t0
                    if first_ts is None:
                        first_ts = total
                    gen = total - first_ts
                    speed = len(full_text) / gen if gen > 0 else 0
                    print(f"  TTFT={first_ts:.2f}s  生成={gen:.2f}s  "
                          f"{speed:.0f}ch/s  total={total:.2f}s")
                    print(f"  回复前80字: {full_text[:80]}")
                    await ws.send(json.dumps({"type": "session.close"}))
                    break
        gpu2 = gpu_info()
        if gpu2:
            print(f"  GPU 显存: {gpu2.get('memory_used_mib', 0):.0f} MiB  "
                  f"利用率: {gpu2.get('gpu_util_pct', 'N/A')}%")
    else:
        print("\n# 测试 2: 图片理解 (跳过, 使用 --image 指定)")

    # ===== 3. 视频理解 =====
    if args.video and os.path.exists(args.video):
        print(f"\n# 测试 3: 视频理解 ({args.video})")
        with open(args.video, "rb") as f:
            vid_b64 = base64.b64encode(f.read()).decode()
        async with websockets.connect(BACKEND_URL, max_size=128*1024*1024) as ws:
            init = {"type": "session.init", "payload": {"mode": "turn_based", "use_tts": False}}
            await ws.send(json.dumps(init))
            await ws.recv()

            append = {
                "type": "input.append",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "video", "data": vid_b64},
                            {"type": "text", "text": video_benchmark_text()},
                        ],
                    }],
                    "streaming": True,
                    "generation": {"max_new_tokens": 256},
                },
            }
            t0 = time.time()
            print(f"  发送视频 ({len(vid_b64)//1024} KB base64)...")
            await ws.send(json.dumps(append))
            full_text = ""
            first_ts = None
            while True:
                msg = json.loads(await ws.recv())
                t = msg.get("type")
                if t == "response.output.delta":
                    chunk = msg.get("text", "")
                    if first_ts is None and chunk:
                        first_ts = time.time() - t0
                    full_text += chunk
                elif t == "response.done":
                    total = time.time() - t0
                    if first_ts is None:
                        first_ts = total
                    gen = total - first_ts
                    speed = len(full_text) / gen if gen > 0 else 0
                    print(f"  TTFT={first_ts:.2f}s  生成={gen:.2f}s  "
                          f"{speed:.0f}ch/s  total={total:.2f}s")
                    print(f"  回复前80字: {full_text[:80]}")
                    await ws.send(json.dumps({"type": "session.close"}))
                    break
        gpu3 = gpu_info()
        if gpu3:
            print(f"  GPU 显存: {gpu3.get('memory_used_mib', 0):.0f} MiB  "
                  f"利用率: {gpu3.get('gpu_util_pct', 'N/A')}%")
    else:
        print("\n# 测试 3: 视频理解 (跳过, 使用 --video 指定)")

    # ===== GPU 峰值汇总 =====
    gpu_final = gpu_info()
    if gpu_final:
        print(f"\n{'='*60}")
        print(f"GPU 峰值显存: {gpu_final.get('memory_used_mib', 0):.0f} MiB")
        print(f"GPU 峰值利用率: {gpu_final.get('gpu_util_pct', 'N/A')}%")

    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
