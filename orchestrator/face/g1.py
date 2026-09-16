"""G1 人脸库的 ctypes 绑定。

对应头文件 ``board-face-and-cloud-infer/G1/include/sdk_stream.h``。

**没有回调** —— ``create`` 一次，每帧调一次 ``feed_*``，**当场拿结果**。
跟踪/说话/唤醒在 C++ 内部跨帧保持。

**实测验证过的三个输出**（G1/sample 回放）：

    f=66  face_valid=1 score=0.88  speaking=0 lip=SILENT  interact=0
    f=84  face_valid=1 score=0.89  speaking=0 lip=SILENT  interact=1  ← 唤醒
    f=119 face_valid=1 score=0.84  speaking=0 lip=SILENT  interact=1

⚠️ 生产必须 ``G1_FACE_DEBUG=0``，否则 ``create`` 就写 debug 到
``~/G1/debug/sess_*``（视频最多约 4GB）。
"""
from __future__ import annotations

import ctypes
import logging
import os
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int64, c_uint8, c_void_p
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

G1_IDENTIFY_SIZE = 512
G1_IDENTIFY_ELEMS = G1_IDENTIFY_SIZE * G1_IDENTIFY_SIZE * 3   # 786432
G1_IDENTIFY_FRAMES = 12


class G1FaceBox(ctypes.Structure):
    _fields_ = [
        ("valid", c_int),
        ("left", c_float),
        ("top", c_float),
        ("right", c_float),
        ("bottom", c_float),
        ("score", c_float),
    ]


class G1FaceState(ctypes.Structure):
    """每帧 state（``sdk_stream.h`` 的 ``G1FaceState``）。

    ⚠️ 字段顺序必须与头文件**逐字一致** —— 顺序错了不会报错，只会读到垃圾值。

    **刷新节奏**：心跳绑在 landmark 上（``lip_detect_every=5``，≈208ms 出一次），
    离散状态（``track_id`` / 两个把握档）变化时**立即**刷新。所以对调用方来说
    state 大约每 5 帧来一次，用 ``state_seq`` 判断「本次是不是新的」。

    ⚠️ ``identity_id`` / ``identity_confidence`` / ``display_name`` 由 C 侧每帧
    **清空**，规范做法是 Python 识别层在 ``feed`` 返回后写回。本绑定不写回，
    身份改由 ``to_observation`` 从本仓库自己的识别流水线填（见该函数）。
    """

    _fields_ = [
        ("face_present_confidence", ctypes.c_char * 8),
        ("lip_speaking_confidence", ctypes.c_char * 8),
        ("track_id", c_int64),
        ("dwell_ms", c_int64),
        ("bbox_area_ratio", c_float),
        ("slot", c_int64),
        ("state_seq", c_int64),
        ("timestamp_us", c_int64),
        ("identity_id", ctypes.c_char * 64),
        ("identity_confidence", ctypes.c_char * 8),
        ("display_name", ctypes.c_char * 64),
        ("identity_source_timestamp_us", c_int64),
        ("frame_index", c_int64),
    ]


class G1FaceResult(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", c_int64),
        ("face", G1FaceBox),
        ("candidate", G1FaceBox),
        ("candidate_valid", c_int),
        ("mar", c_float),
        ("lip_cx", c_float),
        ("lip_cy", c_float),
        ("speaking", c_int),
        ("lip_still", c_int),
        ("lip_state", ctypes.c_char * 16),
        ("reference_valid", c_int),
        ("interacting", c_int),
        # ⚠️ ``state`` 是**追加在末尾**的。C 侧每帧都会写它（216 字节），
        # 如果这里漏声明，C 会直接写穿 Python 分配的缓冲区 → 堆破坏。
        # 下面的 _SIZEOF 自检就是防这个。
        ("state", G1FaceState),
    ]


#: 期望的 ``sizeof(G1FaceResult)``（x86-64）。C 侧加字段而这里没跟上时立刻报错，
#: 而不是等到某次随机崩溃。
_SIZEOF_G1FACE_RESULT = 320

if ctypes.sizeof(G1FaceResult) != _SIZEOF_G1FACE_RESULT:  # pragma: no cover
    raise RuntimeError(
        "G1FaceResult 绑定与 libsdk_stream.so 不匹配："
        f"Python 侧 {ctypes.sizeof(G1FaceResult)} 字节，期望 {_SIZEOF_G1FACE_RESULT}。"
        "多半是 C 侧结构体加了字段 —— 请同步更新本文件的 _fields_（顺序必须一致）。"
    )


class G1Face:
    """一个人脸处理句柄。**单线程使用**（README：``feed`` 单线程）。"""

    def __init__(self, lib_path: str, model_dir: str,
                 identify_enabled: bool = False,
                 debug_enabled: bool = False) -> None:
        self.lib_path = lib_path
        self.model_dir = model_dir
        self.handle: Optional[int] = None
        self.identify_enabled = identify_enabled

        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"G1 库不存在: {lib_path}")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        # debug 必须显式关闭（否则写最多 4GB 视频到 ~/G1/debug）
        os.environ.setdefault("G1_FACE_DEBUG", "1" if debug_enabled else "0")

        self.lib = ctypes.CDLL(lib_path)
        self._bind()
        self.handle = self.lib.g1_face_create(model_dir.encode("utf-8"))
        if not self.handle:
            raise RuntimeError(f"g1_face_create 失败（model_dir={model_dir}）")
        if identify_enabled:
            rc = self.lib.g1_face_set_identify_enabled(self.handle, 1)
            logger.info("G1 身份识别已开启（rc=%d）", rc)
        if not debug_enabled:
            try:
                self.lib.g1_face_set_debug_enabled(self.handle, 0)
            except AttributeError:
                pass
        logger.info("G1 初始化完成: %s（models=%s）", lib_path, model_dir)

    # ------------------------------------------------------------------ #

    def _bind(self) -> None:
        L = self.lib
        U8P = POINTER(c_uint8)
        I64P = POINTER(c_int64)
        FLP = POINTER(c_float)

        L.g1_face_create.restype = c_void_p
        L.g1_face_create.argtypes = [c_char_p]

        L.g1_face_feed_bgr.restype = c_int
        L.g1_face_feed_bgr.argtypes = [
            c_void_p, U8P, c_int, c_int, c_int64,
            POINTER(G1FaceResult), FLP,
        ]

        L.g1_face_feed_mjpeg.restype = c_int
        L.g1_face_feed_mjpeg.argtypes = [
            c_void_p, U8P, c_int, c_int64, POINTER(G1FaceResult), FLP,
        ]

        L.g1_face_reset.restype = c_int
        L.g1_face_reset.argtypes = [c_void_p]

        L.g1_face_destroy.restype = c_int
        L.g1_face_destroy.argtypes = [c_void_p]

        # 身份识别 / 唇图 / debug（老库可能缺符号）
        for name in ("g1_face_set_identify_enabled", "g1_face_take_identify_burst",
                     "g1_face_set_person_id", "g1_face_get_person_id",
                     "g1_face_set_debug_enabled", "g1_face_set_lip_ref_enabled",
                     "g1_face_debug_dir"):
            if not hasattr(L, name):
                logger.warning("G1 库缺少符号 %s（相关功能不可用）", name)

        if hasattr(L, "g1_face_set_lip_ref_enabled"):
            L.g1_face_set_lip_ref_enabled.restype = c_int
            L.g1_face_set_lip_ref_enabled.argtypes = [c_void_p, c_int]
        if hasattr(L, "g1_face_debug_dir"):
            # 返回 const char*（未开 debug 时为 NULL）
            L.g1_face_debug_dir.restype = c_char_p
            L.g1_face_debug_dir.argtypes = [c_void_p]

        if hasattr(L, "g1_face_set_identify_enabled"):
            L.g1_face_set_identify_enabled.restype = c_int
            L.g1_face_set_identify_enabled.argtypes = [c_void_p, c_int]
        if hasattr(L, "g1_face_take_identify_burst"):
            L.g1_face_take_identify_burst.restype = c_int
            L.g1_face_take_identify_burst.argtypes = [
                c_void_p, U8P, I64P, c_int,
            ]
        if hasattr(L, "g1_face_set_person_id"):
            L.g1_face_set_person_id.restype = c_int
            L.g1_face_set_person_id.argtypes = [c_void_p, c_int64, c_int64]
        if hasattr(L, "g1_face_get_person_id"):
            L.g1_face_get_person_id.restype = c_int64
            L.g1_face_get_person_id.argtypes = [c_void_p]
        if hasattr(L, "g1_face_set_debug_enabled"):
            L.g1_face_set_debug_enabled.restype = c_int
            L.g1_face_set_debug_enabled.argtypes = [c_void_p, c_int]

    # ------------------------------------------------------------------ #

    def feed_mjpeg(self, jpeg: bytes, ts_us: int) -> tuple:
        """喂一帧 JPEG。返回 ``(rc, G1FaceResult)``。

        rc: 0 成功；-2 = JPEG 解码失败。
        """
        if self.handle is None:
            raise RuntimeError("句柄已销毁")
        buf = (c_uint8 * len(jpeg)).from_buffer_copy(jpeg)
        out = G1FaceResult()
        rc = self.lib.g1_face_feed_mjpeg(
            self.handle, buf, len(jpeg), ts_us, byref(out), None
        )
        return rc, out

    def feed_bgr(self, bgr: np.ndarray, ts_us: int) -> tuple:
        """喂一帧 BGR（连续 uint8，HWC）。返回 ``(rc, G1FaceResult)``。"""
        if self.handle is None:
            raise RuntimeError("句柄已销毁")
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"需要 (H,W,3) uint8，得到 {bgr.shape} {bgr.dtype}")
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        ptr = bgr.ctypes.data_as(POINTER(c_uint8))
        out = G1FaceResult()
        rc = self.lib.g1_face_feed_bgr(
            self.handle, ptr, w, h, ts_us, byref(out), None
        )
        return rc, out

    def take_identify_burst(self, max_frames: int = G1_IDENTIFY_FRAMES) -> tuple:
        """取走已攒齐的一批 512×512 BGR。

        返回 ``(n, frames_uint8, ts)``。``n=0`` 表示尚无完整批。
        n>0 后**必须**调 ``set_person_id`` 回写结果。
        """
        if self.handle is None or not self.identify_enabled:
            return 0, None, None
        total = max_frames * G1_IDENTIFY_ELEMS
        buf = (c_uint8 * total)()
        ts = (c_int64 * max_frames)()
        n = self.lib.g1_face_take_identify_burst(self.handle, buf, ts, max_frames)
        if n <= 0:
            return 0, None, None
        arr = np.frombuffer(bytes(buf), dtype=np.uint8)
        frames = arr[: n * G1_IDENTIFY_ELEMS].reshape(
            n, G1_IDENTIFY_SIZE, G1_IDENTIFY_SIZE, 3
        )
        return n, frames, [int(ts[i]) for i in range(n)]

    def set_person_id(self, person_id: int, ts_us: int) -> int:
        """回写身份识别结果。>0 成功；-2 = 当前无跟踪目标。"""
        if self.handle is None or not hasattr(self.lib, "g1_face_set_person_id"):
            return -1
        return int(self.lib.g1_face_set_person_id(self.handle, person_id, ts_us))

    def get_person_id(self) -> int:
        if self.handle is None or not hasattr(self.lib, "g1_face_get_person_id"):
            return -1
        return int(self.lib.g1_face_get_person_id(self.handle))

    def reset(self) -> None:
        """清跟踪/说话/唤醒状态，不卸载模型（人走了再进来时用）。"""
        if self.handle is not None:
            self.lib.g1_face_reset(self.handle)

    def close(self) -> None:
        if self.handle is not None:
            try:
                self.lib.g1_face_destroy(self.handle)
            finally:
                self.handle = None
                logger.info("G1 已销毁")


# ---------------------------------------------------------------------- #
#  结果 → 观测的转换
# ---------------------------------------------------------------------- #

#: 身份相似度 → 档位的高分界。与 G1 库 ``g1face.runtime.IDENTITY_HIGH_SIM``
#: 以及 ``G1/接口文档.md`` 的口径一致：<阈值 LOW、阈值~0.50 MEDIUM、>=0.50 HIGH。
IDENTITY_HIGH_SIM = 0.50


def identity_confidence_from_similarity(similarity, threshold: float,
                                        high_sim: float = None) -> str:
    """相似度 → ``NONE`` / ``LOW`` / ``MEDIUM`` / ``HIGH``。

    口径抄自 G1 库自身（``g1face/runtime.py``），保持一致；``high_sim`` 被调到
    阈值之下时夹到阈值，避免 MEDIUM 出现空档。
    """
    if similarity is None:
        return "NONE"
    hi = IDENTITY_HIGH_SIM if high_sim is None else float(high_sim)
    hi = max(hi, float(threshold))
    sim = float(similarity)
    if sim < threshold:
        return "LOW"
    if sim < hi:
        return "MEDIUM"
    return "HIGH"


def _text(raw) -> Optional[str]:
    """C 侧 ``char[]`` → str；空串归一成 ``None``（协议里是 null，不是空串）。"""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    s = str(raw or "").rstrip("\x00").strip()
    return s or None


def _token(raw) -> str:
    """C 侧档位 ``char[]`` → 大写 token；空 / 未知一律 ``NONE``。"""
    s = (_text(raw) or "").upper()
    return s if s in ("HIGH", "MEDIUM", "LOW", "NONE") else "NONE"


def to_observation(res: G1FaceResult, t: int,
                   person_id: int = -1,
                   identity: Optional[dict] = None) -> "FaceObservation":  # noqa: F821
    """把 C 结构体转成 Python 观测对象。

    ⚠️ ``G1FaceResult`` **不含** person_id —— 它由 ``g1_face_get_person_id()``
    单独查询（调用方传入，避免每帧多一次 C 调用）。

    ``identity`` 是本仓库识别流水线的结果 ``{uid, name, similarity}``（或 None）。
    ⚠️ C 侧每帧都把 ``state`` 里的身份三元组**清空**（规范做法是 Python 识别层
    写回），所以这里不从 C 读身份，而是用自己流水线的结果填 —— 口径用
    :func:`identity_confidence_from_similarity`，与 G1 库一致。
    """
    from .signals import FaceObservation

    f = res.face
    lip = res.lip_state.decode("utf-8", "ignore").rstrip("\x00") or "SILENT"
    st = res.state

    # 身份三元组：有识别结果才给，失败不留错误名字（与库的约束一致）。
    identity_id = display_name = None
    identity_conf = "NONE"
    if identity and identity.get("uid"):
        identity_id = identity.get("uid")
        sim = identity.get("similarity")
        identity_conf = identity_confidence_from_similarity(sim, 0.36)
        # 只有 MEDIUM 及以上才给可称呼的名字
        if identity_conf in ("MEDIUM", "HIGH"):
            display_name = identity.get("name")

    track_id = int(st.track_id)
    state = {
        # 定性档位
        "face_present_confidence": _token(st.face_present_confidence),
        "lip_speaking_confidence": _token(st.lip_speaking_confidence),
        # 跟踪
        "track_id": track_id if track_id >= 0 else None,
        "dwell_ms": int(st.dwell_ms),
        "bbox_area_ratio": round(float(st.bbox_area_ratio), 4),
        # 身份（来自本仓库识别流水线，不是 C）
        "identity_id": identity_id,
        "identity_confidence": identity_conf,
        "display_name": display_name,
        # 诊断
        "slot": int(st.slot),
        "state_seq": int(st.state_seq),
        "frame_index": int(st.frame_index),
    }

    return FaceObservation(
        t=t,
        valid=bool(f.valid),
        box=(float(f.left), float(f.top), float(f.right), float(f.bottom))
        if f.valid else None,
        score=float(f.score),
        speaking=bool(res.speaking),
        lip_state=lip,
        interacting=bool(res.interacting),
        person_id=int(person_id),
        state=state,
        state_seq=int(st.state_seq),
    )
