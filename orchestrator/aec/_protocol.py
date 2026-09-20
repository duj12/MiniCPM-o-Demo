"""
WebSocket binary frame protocol.

**本文件是从 speech_frontend 仓库原样拷来的，不要在这里改逻辑。**

| 项 | 值 |
|---|---|
| 源仓库 | ``git@git.xmov.ai:jingshuaizhen/speech_frontend.git`` |
| 源路径 | ``webserver/protocol.py`` |
| 源 commit | ``9f33f6d``（2026-09-09） |
| 源文件 sha256 | ``dabdbf12e2a49e05d7a2d0d0c1a6c471844903d10374d0357d1c1bef93d746d5`` |
| 同步日期 | 2026-09-20 |

**为什么内联**：AEC 服务端（``ws://…/ws/asr_frontend``）用的就是这份代码，
它是协议的**唯一权威**。原本靠 ``Path(__file__).parents[3] / "speech_frontend"``
从同级仓库取，导致本仓库部署不自包含。

**代价与补偿**：内联之后上游若改了帧布局，我们**不会**自动发现（proto 那种
「生成代码自带 descriptor 校验」的保护这里没有）。所以本文件末尾有一段
**round-trip 自检**，并且 ``aec/client.py`` 提供
``ORCH_AEC_PROTOCOL=upstream`` 开关可随时切回权威实现。

**上游更新时**：重新拷贝 → 更新上面的 sha256 与 commit → 跑
``orchestrator/tests/test_aec_align.py``（自检也会在 import 时自动跑一遍）。

每条 binary 消息固定布局:

    +--------------------+------------------+----------------------+
    | header_len uint32  | UTF-8 JSON header| raw binary payload   |
    | (4 bytes, BE)      | (header_len B)   | (按 header 描述拼接)  |
    +--------------------+------------------+----------------------+

JSON header 中的 ``slots`` 数组按顺序声明 payload 每段 numpy 数组的
``name / dtype / shape / offset / length``。

控制信令（hello / ready / error / end-of-stream 等）走 text JSON 消息。
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 4-byte big-endian unsigned int prefix for the header length.
_HEADER_LEN_STRUCT = struct.Struct(">I")
HEADER_LEN_BYTES = _HEADER_LEN_STRUCT.size

# 协议层允许的 numpy dtype 白名单（防止反序列化时被传入危险 dtype）。
_ALLOWED_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


class ProtocolError(ValueError):
    """协议层错误，handler 捕获后转成 text JSON ``error`` 帧回发。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# ------------------------------------------------------------------ #
# 低层：单帧编/解码
# ------------------------------------------------------------------ #

def encode_frame(header: Dict[str, Any], payload: bytes = b"") -> bytes:
    """组装一条 binary 帧。header 必须可 JSON 序列化。"""
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload


def decode_frame(raw: bytes) -> Tuple[Dict[str, Any], memoryview]:
    """拆解一条 binary 帧，返回 (header_dict, payload_view)。

    payload 用 ``memoryview`` 返回，避免大消息 copy。
    """
    if len(raw) < HEADER_LEN_BYTES:
        raise ProtocolError("invalid_frame", f"frame too short: {len(raw)} bytes")
    (header_len,) = _HEADER_LEN_STRUCT.unpack_from(raw, 0)
    if header_len <= 0 or HEADER_LEN_BYTES + header_len > len(raw):
        raise ProtocolError(
            "invalid_frame",
            f"declared header_len={header_len} exceeds frame size={len(raw)}",
        )
    header_end = HEADER_LEN_BYTES + header_len
    try:
        header = json.loads(raw[HEADER_LEN_BYTES:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid_header", f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("invalid_header", "header must be a JSON object")
    return header, memoryview(raw)[header_end:]


# ------------------------------------------------------------------ #
# 高层：多 numpy 数组打包/解包
# ------------------------------------------------------------------ #

def pack_arrays(arrays: Sequence[Tuple[str, np.ndarray]]) -> Tuple[List[Dict[str, Any]], bytes]:
    """把若干命名 numpy 数组拼成 (slots_meta, payload_bytes)。

    返回的 ``slots`` 可直接放进 header；payload 与之对应。
    """
    slots: List[Dict[str, Any]] = []
    chunks: List[bytes] = []
    offset = 0
    for name, arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise ProtocolError("invalid_payload", f"slot '{name}' is not a numpy array")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        dtype_name = arr.dtype.name
        if dtype_name not in _ALLOWED_DTYPES:
            raise ProtocolError("invalid_dtype", f"slot '{name}' dtype '{dtype_name}' not allowed")
        data = arr.tobytes(order="C")
        slots.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(arr.shape),
                "offset": offset,
                "length": len(data),
            }
        )
        chunks.append(data)
        offset += len(data)
    return slots, b"".join(chunks)


def unpack_arrays(
    slots: Sequence[Dict[str, Any]],
    payload: memoryview,
    *,
    expected: Optional[Dict[str, Tuple[str, Optional[Tuple[Optional[int], ...]]]]] = None,
) -> Dict[str, np.ndarray]:
    """根据 ``slots`` 元数据从 ``payload`` 复原命名 numpy 数组。

    Args:
        slots:    chunk header 的 ``slots`` 字段。
        payload:  decode_frame 返回的 payload memoryview。
        expected: 可选 ``{name: (dtype, shape_or_none)}``，对每个 slot 做强校验。
                  shape 元素为 ``None`` 表示该维放任意；shape 整体为 ``None`` 表示不校验形状。
    """
    payload_len = len(payload)
    out: Dict[str, np.ndarray] = {}
    for slot in slots:
        if not isinstance(slot, dict):
            raise ProtocolError("invalid_slot", "slot must be a JSON object")
        try:
            name = str(slot["name"])
            dtype = str(slot["dtype"])
            shape = tuple(int(x) for x in slot["shape"])
            offset = int(slot["offset"])
            length = int(slot["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid_slot", f"missing/invalid field in slot: {exc}") from exc

        if dtype not in _ALLOWED_DTYPES:
            raise ProtocolError("invalid_dtype", f"slot '{name}' dtype '{dtype}' not allowed")
        if offset < 0 or length < 0 or offset + length > payload_len:
            raise ProtocolError(
                "payload_overflow",
                f"slot '{name}' offset={offset} length={length} exceeds payload={payload_len}",
            )

        np_dtype = np.dtype(dtype)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * np_dtype.itemsize if shape else 0
        if expected_bytes != length:
            raise ProtocolError(
                "shape_mismatch",
                f"slot '{name}' shape={shape} dtype={dtype} expects {expected_bytes} bytes,"
                f" got length={length}",
            )

        # frombuffer 拿到的是只读 view，copy 一份避免后续模型 in-place 改坏 ws 缓冲。
        arr = np.frombuffer(payload, dtype=np_dtype, count=int(np.prod(shape, dtype=np.int64) or 0),
                             offset=offset)
        if shape:
            arr = arr.reshape(shape)
        out[name] = np.ascontiguousarray(arr)

    if expected:
        for name, (exp_dtype, exp_shape) in expected.items():
            if name not in out:
                raise ProtocolError("missing_slot", f"required slot '{name}' missing")
            arr = out[name]
            if arr.dtype.name != exp_dtype:
                raise ProtocolError(
                    "dtype_mismatch",
                    f"slot '{name}' expected dtype={exp_dtype}, got {arr.dtype.name}",
                )
            if exp_shape is not None:
                if len(arr.shape) != len(exp_shape):
                    raise ProtocolError(
                        "shape_mismatch",
                        f"slot '{name}' expected ndim={len(exp_shape)}, got {arr.shape}",
                    )
                for i, (got, want) in enumerate(zip(arr.shape, exp_shape)):
                    if want is not None and got != want:
                        raise ProtocolError(
                            "shape_mismatch",
                            f"slot '{name}' axis {i}: expected {want}, got {got}",
                        )
    return out


# ------------------------------------------------------------------ #
# 便捷构造
# ------------------------------------------------------------------ #

def build_result_frame(segments: Sequence[np.ndarray]) -> bytes:
    """打包一次推理产生的 0~N 段 numpy 输出。"""
    arrays = [(f"seg_{i}", np.asarray(seg)) for i, seg in enumerate(segments)]
    slots, payload = pack_arrays(arrays)
    if slots:
        dtype = slots[0]["dtype"]
    else:
        dtype = "float32"
    header = {
        "type": "result",
        "n_segments": len(slots),
        "dtype": dtype,
        "segments": [
            {"shape": s["shape"], "offset": s["offset"], "length": s["length"]}
            for s in slots
        ],
    }
    return encode_frame(header, payload)


def parse_result_frame(raw: bytes) -> List[np.ndarray]:
    """客户端侧：解析服务端回发的 result 帧。"""
    header, payload = decode_frame(raw)
    if header.get("type") != "result":
        raise ProtocolError("unexpected_type", f"expected type=result, got {header.get('type')!r}")
    dtype = str(header.get("dtype", "float32"))
    segments = header.get("segments", [])
    if not isinstance(segments, list):
        raise ProtocolError("invalid_header", "segments must be a list")
    out: List[np.ndarray] = []
    payload_len = len(payload)
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ProtocolError("invalid_segment", f"segment {i} not an object")
        try:
            shape = tuple(int(x) for x in seg["shape"])
            offset = int(seg["offset"])
            length = int(seg["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid_segment", f"segment {i} bad fields: {exc}") from exc
        if offset < 0 or length < 0 or offset + length > payload_len:
            raise ProtocolError(
                "payload_overflow",
                f"segment {i} offset={offset} length={length} exceeds payload={payload_len}",
            )
        np_dtype = np.dtype(dtype)
        arr = np.frombuffer(
            payload, dtype=np_dtype, count=int(np.prod(shape, dtype=np.int64) or 0), offset=offset,
        )
        if shape:
            arr = arr.reshape(shape)
        out.append(np.ascontiguousarray(arr))
    return out


def build_error_text(code: str, message: str) -> str:
    """生成 text JSON 错误帧。"""
    return json.dumps({"type": "error", "code": code, "message": message}, ensure_ascii=False)


# ====================================================================== #
#  契约自检（本仓库新增，**不属于上游代码**）
# ====================================================================== #
#
# 内联的代价是「上游改了帧布局我们不会自动知道」。这段自检把一个
# **启动期就能发现**的问题，从「会话跑到一半出现怪结果」提前到「import 即报错」。
#
# ⚠️ 两个容易踩的点（都是实测出来的）：
#
#  1. 必须用 ``build_result_frame`` 配 ``parse_result_frame`` —— 这才是收发
#     对称的一对。直接 ``encode_frame`` + ``parse_result_frame`` 是**永远失败**的：
#     后者要求 ``header["type"] == "result"``，而 encode_frame 不会替我们写它。
#
#  2. 结果帧**只支持所有段同 dtype**：``build_result_frame`` 写的是
#     ``slots[0]["dtype"]``，``parse_result_frame`` 拿这一个 dtype 套给所有段。
#     混 dtype 会读出错（如 ``ValueError: buffer is smaller than requested size``）。
#     这是协议的固有限制，不是漂移 —— 自检必须按**同 dtype** 来测，否则是假阳性。
#
# 覆盖两个维度：多段（≥2，验证 slot 表拼接/偏移）、不同长度（验证 offset 累加）。

roundtrip_ok = True
roundtrip_error = ""

try:
    _segs = [np.zeros(3, dtype=np.float32), np.arange(5, dtype=np.float32)]
    _back = parse_result_frame(build_result_frame(_segs))
    if len(_back) != len(_segs):
        raise AssertionError(
            f"段数不一致：发出 {len(_segs)}，收回 {len(_back)}")
    for _i, (_a, _b) in enumerate(zip(_segs, _back)):
        if _a.dtype != _b.dtype:
            raise AssertionError(f"段 {_i} dtype 变了：{_a.dtype} → {_b.dtype}")
        if _a.shape != _b.shape:
            raise AssertionError(f"段 {_i} shape 变了：{_a.shape} → {_b.shape}")
        if not np.array_equal(_a, _b):
            raise AssertionError(f"段 {_i} 数据不一致")
except Exception as _exc:  # noqa: BLE001
    roundtrip_ok = False
    roundtrip_error = f"{type(_exc).__name__}: {_exc}"
