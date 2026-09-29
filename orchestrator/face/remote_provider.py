"""远端人脸 provider —— 走 105 上已部署的 G1 人脸服务（HTTP），不加载 `.so`。

对应服务：``board-face-and-cloud-infer/G1/face_service_server.py``
（105 部署在 ``http://192.168.89.105:8767``，见该仓库的 ``服务使用与更新.md``）。

## 为什么要能远端跑

本地 CDLL 那条路（``g1face_provider.py``）要求 ``libsdk_stream.so`` 与调用方的
OpenCV / libstdc++ / libffi 三者版本匹配，**二进制不通用** —— 换机器就得在目标机
重编。实测为让 105 跑起来花了数小时处理这三类环境问题。服务化之后调用方只需一个
URL。

## 两种模式并存（不是替换）

设备端 / 机器人上**本地直调延迟更低**（省一次网络往返 + JPEG 编解码），仍然走
``G1FaceProvider``。只有云端 / 测试环境走这里。分支在 ``main.build_face()``。

## 与 ``G1FaceProvider`` 的接口完全一致

``FaceWorker`` 一行都不用改（它只认鸭子类型）：

    process(jpeg, t) -> FaceObservation | None
    poll_identity(t) -> IdentityEvent | None
    close() / _frame_size / available / wake_dwell_ms

## ⚠️ 两个时钟不是一回事

``process`` 的入参 ``t`` 是**会话的音频采样轴**（16kHz 采样序号，
``session.on_video_face`` 传的是 ``clock.now()``），而 HTTP 请求要的是**单调微秒
时间戳**。两者必须分开，理由见 ``_next_timestamp_us``。
"""
from __future__ import annotations

import base64
import http.client
import json
import logging
import socket
import time
from typing import Any, Optional, Tuple
from urllib.parse import urlsplit

from .signals import FaceObservation, IdentityEvent

logger = logging.getLogger(__name__)

#: 服务端默认上限（``G1_FACE_MAX_JPEG_BYTES``）。超了在本地就丢掉，
#: 别浪费上行 —— 服务端也会以 413 拒掉。
_DEFAULT_MAX_JPEG_BYTES = 8 * 1024 * 1024

#: 与本地 provider 一致：``recognized`` 命中已有条目，``enrolled`` 是在线注册。
_IN_GALLERY = ("recognized", "enrolled")

#: 初始的「无身份」键 —— 避免第一帧就发一条空身份事件（照抄本地实现）。
_NO_IDENT = (None, "NONE", None, None)


class RemoteFaceError(RuntimeError):
    """服务端返回了非 2xx。``status`` 为 None 表示是连接层失败。"""

    def __init__(self, status: Optional[int], detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}" if status else detail)
        self.status = status
        self.detail = detail


class RemoteFaceProvider:
    """105 人脸服务的适配层（HTTP ``POST /face/detect``）。

    **单线程使用** —— 由 ``FaceWorker`` 的专用线程保证（与服务端的
    「同一 session 的帧必须按序发送」要求一致）。
    """

    def __init__(self, base_url: str, *, session_id: str,
                 wake_dwell_ms: int = 2000,
                 timeout_s: float = 2.0,
                 max_jpeg_bytes: int = _DEFAULT_MAX_JPEG_BYTES,
                 close_timeout_s: float = 1.0) -> None:
        parts = urlsplit(base_url if "://" in base_url else f"http://{base_url}")
        if not parts.hostname:
            raise ValueError(f"远端人脸服务地址非法: {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self._host = parts.hostname
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        #: URL 可能带路径前缀（反代），保留下来拼在端点前面
        self._prefix = parts.path.rstrip("/")
        self.session_id = session_id
        self.wake_dwell_ms = int(wake_dwell_ms)
        self._timeout_s = float(timeout_s)
        self._close_timeout_s = float(close_timeout_s)
        self._max_jpeg_bytes = int(max_jpeg_bytes)

        self._conn: Optional[http.client.HTTPConnection] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        #: 单调微秒的上一帧值 —— 保证永不倒退（服务端 < 上一次会回 409）
        self._last_ts_us = 0
        self._pending_ident: Optional[IdentityEvent] = None
        self._last_ident_key = _NO_IDENT
        self._cur_t = 0
        self._last_state: Optional[dict] = None

        #: 诊断计数（``_healthy`` 只用于建会话日志与测试门槛，不进控制流）
        self._healthy = True
        self.frames_sent = 0
        self.frames_failed = 0
        self.failures = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  只读属性（对齐 G1FaceProvider）
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        """服务是否还在正常应答。失败只影响日志，不影响控制流。"""
        return self._healthy

    # ------------------------------------------------------------------ #
    #  连接
    # ------------------------------------------------------------------ #

    def _ensure_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout_s)
        return self._conn

    def _drop_conn(self) -> None:
        """丢掉连接。

        ⚠️ 出错后**必须**丢，不能复用：上一次响应可能只读了一半，
        残留字节会被当成下一次响应的状态行 —— 表现为「之后每一帧都失败」，
        而且看起来像服务端坏了。重连很便宜，别省。
        """
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _post(self, path: str, payload: dict[str, Any],
              timeout: Optional[float] = None) -> dict[str, Any]:
        """POST JSON，返回解析后的 dict。非 2xx 抛 ``RemoteFaceError``。

        ⚠️ **连接层失败会重连重试一次**（服务端已应答的错误**不**重试）。

        为什么必须有这个重试：连接是 keep-alive 复用的，服务端可能已经
        主动关掉了它（空闲超时 / 重启）。下次请求复用这条死连接就得到
        `Broken pipe` / `Connection reset` —— 而**请求根本没发出去**，
        所以重试是安全的。

        实测踩过：`close()` 打 `/face/close` 时正好复用了一条死连接，
        `Broken pipe` 被当成「释放失败」忽略掉 ⇒ **服务端会话没被释放**，
        一直挂到 300s TTL。105 上 `G1_FACE_MAX_SESSIONS=4`，泄漏 4 个之后
        第 5 个会话就吃 503、整场没有人脸（现象是「人脸突然坏了」）。
        """
        last_exc: Optional[BaseException] = None
        for attempt in (1, 2):
            body = json.dumps(payload).encode("utf-8")
            conn = self._ensure_conn()
            old_timeout = conn.timeout
            if timeout is not None:
                conn.timeout = timeout
            try:
                conn.request("POST", f"{self._prefix}{path}", body=body,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                raw = resp.read()      # ⚠️ 必须读完，否则连接无法复用
                status = resp.status
            except (OSError, http.client.HTTPException) as exc:
                # 连接层失败 —— 请求没落地，丢连接重试一次
                last_exc = exc
                self._drop_conn()
                if attempt == 2:
                    raise
                logger.debug("%s 连接层失败（%s），重连重试一次", path, exc)
                continue
            finally:
                conn.timeout = old_timeout

            if status // 100 != 2:
                # 服务端**应答了** —— 这是业务错误，重试没意义
                detail = ""
                try:
                    detail = str(
                        json.loads(raw.decode("utf-8")).get("detail", ""))
                except Exception:  # noqa: BLE001
                    detail = raw[:200].decode("utf-8", "replace")
                raise RemoteFaceError(status, detail)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise RemoteFaceError(
                    status, f"响应不是合法 JSON: {exc}") from exc
        raise RemoteFaceError(None, f"请求 {path} 失败: {last_exc}")

    # ------------------------------------------------------------------ #
    #  每帧
    # ------------------------------------------------------------------ #

    def _next_timestamp_us(self) -> int:
        """取本帧的**单调微秒**时间戳（服务端要求严格递增）。

        ⚠️⚠️ **绝不能拿 ``process`` 的入参 ``t`` 当微秒用。**

        ``t`` 是会话的**音频采样轴**（16kHz 采样序号，``session.on_video_face``
        传的 ``clock.now()``），当微秒用会同时踩两个坑：

          (a) 刻度慢 62.5 倍 ⇒ C 侧按 ts 算出的 ``dwell_ms`` 永远到不了
              唤醒阈值 ⇒ **永不唤醒**。而 ``valid`` 一直是 True，看起来
              「检测正常」，只是死活不迎宾 —— 与「ASR 识别完美却零决策」
              是同一类隐蔽故障，且只在 IC 侧可见（``dwell_ms`` 恒 0
              ⇒ 永远判 passerby）。
          (b) 音频路径一停（用户不出声）``clock.now()`` 就不再推进，而视频
              仍在来 ⇒ 相邻帧 ts 相等；音频时钟换锚点/epoch 时还会**倒退**
              ⇒ 服务端回 409，之后**每一帧都 409**，人脸永久失效。

        所以这里用 ``time.monotonic()``，并额外做一次递增保护（同一微秒内
        到两帧、或系统时钟调整时兜底）。注意与 ``FaceObservation.t`` 无关 ——
        那个字段**必须**保持音频采样轴：``FaceWorker._push_lip`` 用
        ``t // LIP_WINDOW_SAMPLES`` 聚合 100ms 窗，换成微秒会把唇动聚合打碎。
        """
        ts = int(time.monotonic() * 1e6)
        if ts <= self._last_ts_us:
            ts = self._last_ts_us + 1
        return ts

    def process(self, jpeg: bytes, t: int) -> Optional[FaceObservation]:
        """喂一帧 JPEG，返回观测；失败 / 坏帧返回 ``None``（**绝不抛**）。"""
        self._cur_t = t

        if not jpeg:
            return None
        if len(jpeg) > self._max_jpeg_bytes:
            # 本地就拦掉，省一次上行（服务端也会 413）
            self._warn_limited("帧过大(%d字节) 超过上限 %d，已跳过",
                               len(jpeg), self._max_jpeg_bytes)
            return None

        ts_us = self._next_timestamp_us()
        payload = {
            "session_id": self.session_id,
            "timestamp_us": ts_us,
            "frame_jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
        }
        try:
            resp = self._post("/face/detect", payload)
        except (RemoteFaceError, socket.timeout, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            self._on_frame_error(exc)
            return None

        # 成功 —— 只有这时才推进时间戳（失败的那帧没有占用服务端时间轴）
        self._last_ts_us = ts_us
        self.frames_sent += 1
        self._healthy = True

        # 帧尺寸（UI 要把框坐标映射到显示区）—— 复用本地那份纯 Python SOF 解析，
        # 不引 cv2（远端的 OpenCV 在服务端，调用方不该依赖它）
        if self._frame_size is None:
            from .g1face_provider import jpeg_size
            sz = jpeg_size(jpeg)
            if sz is not None:
                self._frame_size = sz
                logger.info("远端人脸输入帧尺寸: %dx%d", sz[0], sz[1])

        return self._to_observation(resp, t)

    # ------------------------------------------------------------------ #

    def _to_observation(self, resp: dict[str, Any], t: int) -> FaceObservation:
        """服务响应 → ``FaceObservation``。"""
        # 服务端的 state **一定非空**（本帧新刷的，或最近一次缓存快照）
        cached = resp.get("state") or {}
        if cached:
            self._last_state = cached
        else:
            cached = self._last_state or {}

        valid = bool(resp.get("valid"))
        dwell_ms = int(resp.get("dwell_ms") or 0)
        #: 唤醒判据与本地**同一条**：服务端明确不返回 ``interacting``
        #: （见 ``include/sdk_stream.h`` 与 ``服务使用与更新.md`` 第 2.1 节），
        #: 由调用方按 ``dwell_ms`` 自判。阈值 2000ms = C 侧 wake_ms_high
        #: = IC 的 passerby 阈值，三处必须同值。
        interacting = valid and dwell_ms >= self.wake_dwell_ms

        tid = cached.get("track_id")
        # ⚠️ ``state`` 只在**刷新帧**透出（≈5Hz）。本地 provider 也是这样。
        #    每帧都给会让 ``session.run_face_signals`` 以 25Hz 向下游投
        #    ``FaceState``（它按「有就给」去重），把下游与 IC 写队列淹掉。
        state = dict(cached) if resp.get("state_fresh") else None
        if state is not None:
            self._maybe_emit_identity(cached)

        box = resp.get("box")
        return FaceObservation(
            t=t,
            valid=valid,
            box=tuple(float(v) for v in box) if (valid and box) else None,
            score=float(resp.get("score") or 0.0),
            speaking=bool(resp.get("speaking")),
            lip_state=resp.get("lip_state") or "SILENT",
            interacting=interacting,
            # ⚠️ dwell_ms / track_id 来自**缓存 state**（服务端顶层只有 dwell_ms，
            #    没有 track_id），所以可能滞后一个 ≈208ms 的心跳。可接受 ——
            #    关键是 ``interacting`` 与这里的 ``dwell_ms`` 取自**同一个数**，
            #    自洽。否则唤醒时会出现「interacting=True 但 dwell_ms=0」，
            #    看起来像「没熬够就唤醒了」（本地实现踩过这个坑）。
            dwell_ms=dwell_ms,
            person_id=int(cached.get("person_id") or -1),
            track_id=str(tid) if tid not in (None, "", -1) else None,
            identity_state=cached.get("identity_state"),
            state=state,
            state_seq=int(cached.get("state_seq") or -1),
        )

    # ------------------------------------------------------------------ #
    #  身份
    # ------------------------------------------------------------------ #

    def _maybe_emit_identity(self, cached: dict[str, Any]) -> None:
        """身份变化 → 攒一条待取事件。

        ⚠️ **服务端每帧都返回 ``identity``**（不像本地 provider 只在有 state
        刷新时才有），所以这里**必须**比对身份键去重 —— 否则每帧都会向下游
        投一条 ``FaceIdentity``（25Hz）。本地实现同样靠比对键去重，语义保持一致。
        """
        key = (cached.get("identity_id"), cached.get("identity_confidence"),
               cached.get("display_name"), cached.get("identity_state"))
        if key == self._last_ident_key:
            return
        self._last_ident_key = key

        uid = cached.get("identity_id") or None
        if not uid:
            # 身份失败 —— 服务端已保证不留错误 display_name，这里不发事件，
            # 由 ``FaceWake(end)`` 那条路径负责清空 IC 侧身份。
            self._pending_ident = None
            return
        istate = cached.get("identity_state")
        logger.info("身份识别（远端）：%s (uid=%s state=%s)",
                    cached.get("display_name") or "-", uid, istate)
        self._pending_ident = IdentityEvent(
            t=self._cur_t,
            track_id=0,
            person_id=int(cached.get("person_id") or -1),
            uid=uid,
            name=cached.get("display_name") or None,
            # 服务端不返回相似度 —— 留 0，别假装有值
            similarity=0.0,
            is_enrolled=istate in _IN_GALLERY,
            identity_state=istate,
        )

    def poll_identity(self, t: int) -> Optional[IdentityEvent]:
        """取走待发身份事件（没有则 ``None``）。"""
        ev, self._pending_ident = self._pending_ident, None
        return ev

    # ------------------------------------------------------------------ #
    #  错误
    # ------------------------------------------------------------------ #

    def _on_frame_error(self, exc: BaseException) -> None:
        self._drop_conn()
        self.frames_failed += 1
        self.failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        status = getattr(exc, "status", None)

        if status == 409:
            # 时间戳倒退 / 会话已被关闭。**不要**去调 /face/reset ——
            # 那会把 dwell 清零，代价比丢这一帧大得多。
            self._warn_limited("远端人脸返回 409（%s）—— 丢弃本帧", exc)
            return
        if status == 503:
            # ``G1_FACE_MAX_SESSIONS`` 打满（105 上是 4）。本会话降级为无人脸，
            # 但继续重试 —— 服务端会按 TTL 回收空闲会话腾出名额。
            self._healthy = False
            self._warn_limited("远端人脸服务会话已满（%s）—— "
                               "本会话降级为无人脸，将持续重试", exc)
            return
        if status is None:
            # 连接层失败（超时 / 连不上）—— 视为服务不可用
            self._healthy = False
        self._warn_limited("远端人脸请求失败（%s）—— 已丢弃本帧", exc)

    def _warn_limited(self, fmt: str, *args: Any) -> None:
        """首次 + 每 100 次 warning，其余 debug。

        沿用 ``interaction/client.py`` 的写法：**持续失败必须看得见，但不能刷屏**。
        早先这类失败被打成 debug，于是「参数/环境不对」导致的持续失败
        完全不可见，排查了很久。
        """
        n = self.failures
        if n <= 1 or n % 100 == 0:
            logger.warning(fmt + ("（第 %d 次）" % n if n > 1 else ""), *args)
        else:
            logger.debug(fmt, *args)

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """释放服务端会话（尽力而为），再丢掉连接。

        服务端还有 300s 空闲 TTL 兜底，所以这里失败**只记日志** ——
        会话收尾不该因为一个人脸服务调用而卡住或报错。
        """
        try:
            self._post("/face/close", {"session_id": self.session_id},
                       timeout=self._close_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.info("远端人脸会话释放失败（可忽略，服务端有 TTL 兜底）：%s", exc)
        self._drop_conn()
        logger.info("远端人脸已关闭: 帧 sent=%d failed=%d",
                    self.frames_sent, self.frames_failed)
