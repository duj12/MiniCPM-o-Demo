"""本地人脸 provider —— 组合 G1 人脸库 + FaceService 身份识别。

分工（对齐 board 文档的角色说明）：

    G1 库      每帧 feed → interacting / speaking / lip_state（跨帧状态在 C++）
               ↓ 唤醒后累计 12 帧 512×512 裁剪人脸 burst
    FaceService  identify(burst) → (uid, name, similarity) 或 None
               ↓ 结果映射成 int64 person_id
    G1 库       set_person_id 回填

⚠️ ``FaceService`` 是**单线程**的（README 明确），且需要加锁 —— 本类
在人脸专用线程里调用它，因此天然串行；但仍加锁以防未来多线程访问。

⚠️ 身份是 string（12 位 hex uid）而 G1 要 int64 person_id，需要一张
映射表。本类用稳定的哈希（uid → 递增 int64）并持久化到 json。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .g1 import G1Face, G1_IDENTIFY_ELEMS, G1_IDENTIFY_SIZE, to_observation
from .signals import FaceObservation, IdentityEvent

logger = logging.getLogger(__name__)


class PersonIdMap:
    """uid（string）↔ person_id（int64）的稳定映射，落盘持久化。

    G1 的 person_id 是 int64，而 FaceService 返回 12 位 hex uid —— 必须有
    映射。用递增整数而非哈希，避免碰撞且便于日志阅读。
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._by_uid: Dict[str, int] = {}
        self._by_id: Dict[int, str] = {}
        self._next = 1
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._by_uid = {str(k): int(v) for k, v in data.get("by_uid", {}).items()}
            self._by_id = {int(v): str(k) for k, v in self._by_uid.items()}
            self._next = max(self._by_id.keys(), default=0) + 1
            logger.info("加载 person_id 映射 %d 条（%s）", len(self._by_uid), self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("person_id 映射加载失败（忽略）: %s", exc)

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"by_uid": self._by_uid}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("person_id 映射保存失败（忽略）: %s", exc)

    def get_or_create(self, uid: str) -> int:
        with self._lock:
            if uid in self._by_uid:
                return self._by_uid[uid]
            pid = self._next
            self._next += 1
            self._by_uid[uid] = pid
            self._by_id[pid] = uid
            self._save()
            logger.info("新 person_id %d ↔ uid=%s", pid, uid)
            return pid


class LocalFaceProvider:
    """G1 逐帧处理 + 唤醒后 identity burst → FaceService。"""

    def __init__(self, lib_path: str, model_dir: str,
                 face_service=None, id_map: Optional[PersonIdMap] = None,
                 identify_enabled: bool = True,
                 identify_cooldown_s: float = 5.0,
                 debug: bool = False) -> None:
        self.g1 = G1Face(lib_path, model_dir,
                         identify_enabled=identify_enabled, debug_enabled=debug)
        self.face_service = face_service
        self.id_map = id_map or PersonIdMap()
        self.identify_cooldown_s = identify_cooldown_s
        self._last_identify_at = 0.0
        self._lock = threading.Lock()
        self.frames_seen = 0
        self.identify_calls = 0
        # 记住当前 person_id，避免每帧都调 C
        self._cached_person_id = -1

    # ------------------------------------------------------------------ #

    def process(self, jpeg: bytes, t: int) -> Optional[FaceObservation]:
        """处理一帧 JPEG。返回观测（无人脸时 valid=False）。"""
        ts_us = int(time.monotonic() * 1e6)
        rc, res = self.g1.feed_mjpeg(jpeg, ts_us)
        if rc != 0:
            if rc == -2:
                logger.debug("JPEG 解码失败（%d 字节）", len(jpeg))
            else:
                logger.warning("g1_face_feed_mjpeg 返回 %d", rc)
            return None
        self.frames_seen += 1
        obs = to_observation(res, t, person_id=self._cached_person_id)
        return obs

    def poll_identity(self, t: int) -> Optional[IdentityEvent]:
        """检查是否有待处理的身份 burst。

        只在唤醒后（``interacting``）且冷却期已过才取批 —— 由 worker 在
        每次观测后调用。
        """
        if self.face_service is None or not self.g1.identify_enabled:
            return None
        now = time.monotonic()
        if now - self._last_identify_at < self.identify_cooldown_s:
            return None

        n, frames, ts = self.g1.take_identify_burst()
        if n <= 0:
            return None

        self._last_identify_at = now
        self.identify_calls += 1
        try:
            with self._lock:
                result = self.face_service.identify(
                    [frames[i] for i in range(n)], n_frames=n, pose_check=True
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FaceService.identify 异常: %s", exc)
            return None

        if result is None:
            logger.info("身份识别：不在库或拒识（%d 帧）", n)
            return IdentityEvent(t=t, person_id=-1, is_enrolled=False)

        uid = result.get("uid")
        name = result.get("name")
        sim = float(result.get("similarity", 0.0))
        pid = self.id_map.get_or_create(uid) if uid else -1
        # 回填给 G1，后续帧的 on_face.person_id 就能看到
        rc = self.g1.set_person_id(pid, int(ts[-1]) if ts else 0)
        self._cached_person_id = pid
        logger.info("身份识别：%s (uid=%s sim=%.3f) → person_id=%d (rc=%d)",
                    name, uid, sim, pid, rc)
        return IdentityEvent(
            t=t, person_id=pid, uid=uid, name=name,
            similarity=sim, is_enrolled=True,
        )

    def close(self) -> None:
        try:
            self.g1.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("g1.close 异常: %s", exc)


def load_face_service(model_dir: str, db_path: str, gpu_id: int = 0,
                      threshold: float = 0.36):
    """加载 FaceService（faceidentification 仓库）。

    返回 ``(service, error)``。任一环节失败时 service 为 None 并给出原因，
    调用方据此降级（只做检测不识别，而不是整体失败）。
    """
    try:
        import sys
        fi_root = Path(__file__).resolve().parents[4] / "faceidentification"
        if not fi_root.is_dir():
            return None, f"faceidentification 仓库不存在: {fi_root}"
        if str(fi_root) not in sys.path:
            sys.path.insert(0, str(fi_root))

        from scripts.common import load_config  # type: ignore
        from src.face_service import FaceService  # type: ignore

        cfg = load_config()
        svc = FaceService(cfg["model"], gpu_id=gpu_id, threshold=threshold)
        stats = svc.load_db(db_path)
        logger.info("FaceService 就绪：%d 人 / %d 模板（阈值 %.2f）",
                    stats.get("n_identities", -1),
                    stats.get("n_features", -1), threshold)
        return svc, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
