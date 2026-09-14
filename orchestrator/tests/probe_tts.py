#!/usr/bin/env python3
"""阶段 0 量测：TTS 服务（gRPC）的行为。

回答：
  1. 整段合成 `inference(is_first=is_last=True)` 是否一次流出 24kHz int16
  2. 首帧延迟 / 总耗时 / 音频时长
  3. CHAR_TIME_MAP 的字级时间戳形状（字幕对齐要用）
  4. `check_input_text` / `get_version` 的可用性（上线前的输入校验）

用法（在 106 上跑，因为需要 grpc + TTS/protos）::

    cd /data/megastore/Projects/DuJing/code
    /home/dujing/miniconda3/envs/py310/bin/python -m MiniCPM-o-Demo.orchestrator.tests.probe_tts

    # 指定文本 / 音色
    ... probe_tts.py --text "你好" --tts-type mltts --speaker-id 17

注意：`tts_type` 与 `speaker_id` 的合法组合见 TTS/tests/grpc_client.py 的
`tts_speaker_id_map_for_test`。默认用 mltts/17（中文，测试里验证过）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# TTS 仓库路径（在 106 上位于 code/TTS）
TTS_ROOT = Path(__file__).resolve().parents[3] / "TTS"
if str(TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TTS_ROOT))

DEFAULT_HOST = "192.168.88.253"
DEFAULT_PORT = 31058
DEFAULT_TEXT = "你好，我是语音助手，现在测试一下语音合成。"


@dataclass
class ProbeResult:
    host: str
    port: int
    text: str
    tts_type: str
    speaker_id: str
    audio: Optional[np.ndarray] = None
    char_time_map: Optional[list] = None
    n_frames: int = 0
    first_frame_at: Optional[float] = None
    total_at: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    version: Optional[str] = None
    check_result: Optional[str] = None

    def summary(self) -> str:
        out = ["=" * 68, f"TTS 量测结果  {self.host}:{self.port}", "-" * 68]
        out.append(f"  文本           : {self.text[:40]!r}{'...' if len(self.text) > 40 else ''}")
        out.append(f"  tts_type/spk   : {self.tts_type} / {self.speaker_id}")
        if self.version:
            out.append(f"  get_version    : {self.version}")
        if self.check_result is not None:
            out.append(f"  check_input_text: {self.check_result}")
        if self.audio is not None:
            n = len(self.audio)
            dur = n / 24000.0
            out.append(f"  PCM 样本数     : {n}   时长 = {dur:.2f}s @24kHz int16")
            out.append(f"  音频 dtype     : {self.audio.dtype}")
            if n:
                out.append(f"  幅度范围       : [{int(self.audio.min())}, {int(self.audio.max())}] "
                           f"rms={float(np.sqrt((self.audio.astype(np.float64) ** 2).mean())):.0f}")
                if self.audio.max() == 0 and self.audio.min() == 0:
                    out.append("    [!] 全零音频 —— 合成失败或 speaker_id 不合法")
        out.append(f"  result 帧数    : {self.n_frames}")
        if self.first_frame_at is not None:
            out.append(f"  首帧延迟       : {self.first_frame_at * 1000:.0f} ms")
        if self.total_at is not None:
            out.append(f"  总耗时         : {self.total_at * 1000:.0f} ms")
        if self.audio is not None and self.total_at and len(self.audio):
            rtf = self.total_at / (len(self.audio) / 24000.0)
            out.append(f"  RTF            : {rtf:.3f}")
            out.append(f"  首帧/总耗时比  : {self.first_frame_at / self.total_at * 100:.0f}%"
                       if self.first_frame_at else "")
        if self.char_time_map is not None:
            out.append(f"  CHAR_TIME_MAP  : {len(self.char_time_map)} 项")
            out.append(f"    样例: {json.dumps(self.char_time_map[:4], ensure_ascii=False)}")
            out.append("    [OK] 字级时间戳可用于字幕对齐")
        else:
            out.append("  CHAR_TIME_MAP  : 无 —— 字幕对齐只能用整段时长估算")
        if self.errors:
            out.append(f"  错误: {self.errors}")
        else:
            out.append("  错误: 无")
        out.append("=" * 68)
        return "\n".join(out)


def run_probe(host: str, port: int, text: str, tts_type: str, speaker_id: str,
              out_wav: Optional[str], timeout: float) -> ProbeResult:
    import grpc
    from protos import tts_pb2, tts_pb2_grpc  # type: ignore

    res = ProbeResult(host=host, port=port, text=text, tts_type=tts_type,
                      speaker_id=speaker_id)
    options = [("grpc.max_receive_message_length", 4605632 * 2)]
    with grpc.insecure_channel(f"{host}:{port}", options=options) as channel:
        stub = tts_pb2_grpc.TTSStub(channel)

        # get_version（非致命）
        try:
            v = stub.get_version(tts_pb2.GetVersionRequest(tts_type=tts_type), timeout=10)
            res.version = str(getattr(v, "version", v))
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"get_version 失败: {type(exc).__name__}: {exc}")

        # check_input_text（非致命）
        try:
            c = stub.check_input_text(tts_pb2.Text(text=text, tts_type=tts_type), timeout=10)
            res.check_result = f"result={c.result} reason={getattr(c, 'reason', '')!r}"
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"check_input_text 失败: {type(exc).__name__}: {exc}")

        # 整段合成
        st = time.perf_counter()
        try:
            it = stub.inference(
                tts_pb2.Text(
                    text=text,
                    tts_type=tts_type,
                    is_first=True,
                    is_last=True,
                    return_sep=False,
                    secondary_style_id=tts_pb2.Text.SECONDARY_STYLE_ID.jiangpin,
                    speaker_id=str(speaker_id),
                ),
                metadata=[("grpc-infer-id", "probe_0")],
                timeout=timeout,
            )
            buf = bytearray()
            for r in it:
                res.n_frames += 1
                if res.first_frame_at is None:
                    res.first_frame_at = time.perf_counter() - st
                if r.data_type == 0:  # AUDIO
                    buf.extend(r.data)
                elif r.data_type == 1:  # CHAR_TIME_MAP
                    try:
                        res.char_time_map = json.loads(r.data)
                    except Exception:  # noqa: BLE001
                        res.char_time_map = None
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"inference 失败: {type(exc).__name__}: {exc}")
            return res
        res.total_at = time.perf_counter() - st
        res.audio = np.frombuffer(bytes(buf), dtype=np.int16)

    if out_wav and res.audio is not None and len(res.audio):
        try:
            import soundfile as sf  # type: ignore
            Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_wav, res.audio, 24000)
            print(f"[写入] {out_wav}")
        except ImportError:
            res.errors.append("soundfile 不可用，未写 wav")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="TTS 服务量测")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--tts-type", default="mltts",
                   help="见 TTS/tests/grpc_client.py 的 tts_speaker_id_map_for_test")
    p.add_argument("--speaker-id", default="17")
    p.add_argument("--out-wav", default=None)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    print(f"连接 {args.host}:{args.port}  tts_type={args.tts_type} spk={args.speaker_id}")
    try:
        res = run_probe(args.host, args.port, args.text, args.tts_type,
                        args.speaker_id, args.out_wav, args.timeout)
    except ImportError as exc:
        print(f"[FATAL] 缺少依赖: {exc}")
        print("        需要 grpc（106 的 py310 有）与 TTS/protos 可 import。")
        raise SystemExit(2)
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(res.summary())


if __name__ == "__main__":
    main()
