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
    ]


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

        # 身份识别（老库可能缺符号）
        for name in ("g1_face_set_identify_enabled", "g1_face_take_identify_burst",
                     "g1_face_set_person_id", "g1_face_get_person_id",
                     "g1_face_set_debug_enabled"):
            if not hasattr(L, name):
                logger.warning("G1 库缺少符号 %s（身份识别可能不可用）", name)

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

def to_observation(res: G1FaceResult, t: int,
                   person_id: int = -1) -> "FaceObservation":  # noqa: F821
    """把 C 结构体转成 Python 观测对象。

    ⚠️ ``G1FaceResult`` **不含** person_id —— 它由 ``g1_face_get_person_id()``
    单独查询（调用方传入，避免每帧多一次 C 调用）。
    """
    from .signals import FaceObservation

    f = res.face
    lip = res.lip_state.decode("utf-8", "ignore").rstrip("\x00") or "SILENT"
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
    )
