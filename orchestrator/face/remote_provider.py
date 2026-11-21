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

#: 待发 8PCM 的块数上限（100ms/块 ⇒ 300 块 = 30s）。正常只有 1~2 块。
_PCM8_MAX_PENDING = 300

#: 客户端一块 mic 音频的字节数：1600 采样 × 8 声道 × int16。
_MIC_BLOCK_BYTES = 1600 * 8 * 2

#: **每次 detect 携带的 PCM 切片**（采样）。16k 下 40ms —— 与 25fps 的
#: 帧间隔完全相同。这条就是「视频节拍 = 人脸节拍」的保证：
#: 8768 要求每次 detect 都带 PCM 且相邻请求首尾相接，于是只有两种合法节奏
#:   · 每来一块音频（100ms）才发一帧 ⇒ 人脸 10Hz（旧实现，见下）；
#:   · 每帧切 40ms 音频发一次  ⇒ 25Hz，音频依然连续（现在）。
#: 供给与需求在 16k 下精确相等：16000 采样/s ÷ 25fps = 640 采样/帧。
#: 反例（s-56886bcbcf10）：视频 1191 帧 / 55s，真正进 8768 的只有 513 帧
#: （≈9.3Hz）—— 没有新音频块的视频帧全被跳过，BlazeFace/Landmark/LipFsm
#: 拿不到设计节拍。
_PCM8_SLICE_SAMPLES = 640
_PCM8_SLICE_BYTES = _PCM8_SLICE_SAMPLES * 8 * 2

#: 积压到 **6 片**（240ms）才启动「一次多取」，排到只剩 1 片余量。
#: 阈值不能太小：音频 100ms 一块、视频 40ms 一帧，块边界处缓冲本来就会
#: 到 3 片（1920 采样），过早排水会让下一帧没音频可发 —— 实测 25fps 被
#: 压到 20Hz。6 片只在视频真的慢于音频（实测 20fps 那档触发）时才排水。
_PCM8_BACKLOG_SLICES = 6

#: 服务端单请求 PCM8 上限 —— **实测 1 秒音频**（16k×1s×8ch×2B = 256000B）。
#: 1000ms → 200，1008ms → `413 PCM8 exceeds service limit`。超过的部分
#: 留在队里下一次发（切在整帧边界，所以仍是连续的）。
_PCM8_MAX_SAMPLES = 16000
_PCM8_MAX_BYTES = _PCM8_MAX_SAMPLES * 8 * 2


def _samples_to_us(n: int, sr: int = 16000) -> int:
    """采样数 → 微秒（**整数**运算，服务端要 int）。

    62.5µs/采样不能写成浮点：pydantic 的 int 字段拿到 ``100000.0``
    可能被拒或截断，而这两个时间戳一旦错位就是 422。
    """
    return (int(n) * 1_000_000) // int(sr)


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
                 close_timeout_s: float = 1.0,
                 pcm8: bool = False) -> None:
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

        #: 8PCM 门控（8768）。开 ⇒ 每次 detect 都必须带 PCM，且时间戳
        #: 走**音频采样轴**而不是墙钟（见 ``set_pcm8`` / ``_send_pcm8``）。
        self.pcm8_enabled = bool(pcm8)
        #: 待发的 8 声道交错 int16 **字节缓冲**（不解交错）。按「每帧切
        #: 40ms 一片」消费，见 ``_send_pcm8``；切成整采样（16 字节）边界。
        self._pcm8_buf = bytearray()
        #: 已交给服务端的 PCM 累计采样数 —— 下一块的起点就是它，
        #: 「块间连续」由这个游标保证（不能拿墙钟/帧率去凑）。
        self._pcm8_cursor = 0
        self._pcm8_warned_empty = False
        self.pcm8_blocks = 0

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

        if self.pcm8_enabled:
            # 8768 门控：时间戳走音频采样轴，**每帧切一片 40ms** 音频发送
            # （不是「有新块才发」—— 那会把视频节拍压到 10Hz）
            return self._send_pcm8(jpeg, t)

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

        self._probe_frame_size(jpeg)
        return self._to_observation(resp, t)

    # ------------------------------------------------------------------ #
    #  8PCM（8768 门控）
    # ------------------------------------------------------------------ #

    def set_pcm8(self, raw: bytes) -> None:
        """登记一块客户端送来的 8 声道交错 int16 PCM（**非阻塞、不拷贝**）。

        由 mic ingest 路径调用（``session.on_audio``），工作线程消费、
        拼进下一次 ``/face/detect``。块与块在**音频采样轴**上连续，
        时间戳一律由本类的游标推出，不看墙钟 —— 8768 要求
        ``pcm8_timestamp_us <= 每个采样时刻 < timestamp_us``，且相邻请求的
        PCM 必须首尾相接，游标是唯一能保证这件事的东西。
        """
        if not self.pcm8_enabled:
            return
        if not raw or len(raw) % 16:
            if not self._pcm8_warned_empty:
                self._pcm8_warned_empty = True
                logger.warning("8PCM 块非法（%d 字节，需 8 声道 int16 对齐）"
                               "—— 已丢弃，后续同类不再提示", len(raw))
            return
        self._pcm8_buf += raw
        # 兜底：视频停了但音频还在送（或人脸线程被拖住）时别无限攒。
        # 丢掉最旧的整块会破坏块间连续 —— 之后服务端会以 422 拒绝，
        # 由 ``_resync_pcm8`` 重建会话兜底。
        cap = _PCM8_MAX_PENDING * _MIC_BLOCK_BYTES
        if len(self._pcm8_buf) > cap:
            del self._pcm8_buf[:len(self._pcm8_buf) - cap]
            self._warn_limited("8PCM 积压超过 %d 块（%.0fs）—— 丢弃最旧块",
                               _PCM8_MAX_PENDING, _PCM8_MAX_PENDING * 0.1)

    def _send_pcm8(self, jpeg: bytes, t: int) -> Optional[FaceObservation]:
        """带 PCM 的 detect：**每帧切一片 40ms**（见 ``_PCM8_SLICE_BYTES``）。

        旧实现「没有新 PCM 就不发」把视频节拍绑死到了 100ms 音频块上
        （detect 只有 ~10Hz）。现在无论音频怎么成块到达，每帧都从缓冲里
        抽 640 采样发一次 —— 视频 25fps ⇒ 人脸 25Hz，音频仍然首尾相接。
        """
        avail = len(self._pcm8_buf)
        if avail < _PCM8_SLICE_BYTES:
            # 还不够一帧的音频：这帧跳过。8768 缺 PCM 会 422，
            # 也不能拿上一块充数（违反「块间连续、不得重复」）。
            return None
        # 正常**只取一片**（25fps 下供给与需求精确相等，多取反而会出现
        # 空帧、把请求数拉到 20Hz）。只有积压 ≥3 片才一次多取 —— 比如
        # 视频实际只跑到 21fps、或人脸线程被拖住，两条轴不能越拉越开。
        take = _PCM8_SLICE_BYTES
        if avail >= _PCM8_BACKLOG_SLICES * _PCM8_SLICE_BYTES:
            take = min(avail - _PCM8_SLICE_BYTES, _PCM8_MAX_BYTES)
        take = min(take, _PCM8_MAX_BYTES)        # 服务端单请求上限 1s
        take -= take % 16                        # 切在整采样边界
        buf = bytes(self._pcm8_buf[:take])
        del self._pcm8_buf[:take]
        n = take // 16                           # 每声道采样数
        start_us = _samples_to_us(self._pcm8_cursor)
        end_us = _samples_to_us(self._pcm8_cursor + n)
        payload = {
            "session_id": self.session_id,
            # 帧时间戳取 PCM 末端的音频时刻：窗口 [start, end) 恰好覆盖本块
            # 全部采样，且严格递增（游标只增），不会越界。
            "timestamp_us": end_us,
            "frame_jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
            "pcm8_base64": base64.b64encode(buf).decode("ascii"),
            "pcm8_timestamp_us": start_us,
        }
        try:
            resp = self._post("/face/detect", payload)
        except (RemoteFaceError, socket.timeout, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            # 失败：把**这一段**放回缓冲头部（不是原始块列表 —— 可能已按
            # 上限切过），和后面的部分一起发，仍然是连续的。
            self._pcm8_buf[:0] = buf
            self._on_frame_error(exc)
            return None

        self._pcm8_cursor += n
        self._last_ts_us = end_us
        self.frames_sent += 1
        self.pcm8_blocks += 1
        self._healthy = True
        self._probe_frame_size(jpeg)
        return self._to_observation(resp, t)

    def _resync_pcm8(self) -> None:
        """PCM 轴被服务端拒了（422）——重开会话，从 0 重新计时。

        8768 是按 ``session_id`` 记账的，块一旦不连续（积压丢弃、客户端
        换块）、或者采样与帧时间对不上，就会一直 422。``/face/close``
        让服务端回收该会话，下一次 detect 重新建一个，游标归零即可自愈。
        代价是 dwell 清零 —— 比「人脸永久失效」便宜得多。
        """
        self._pcm8_buf.clear()
        self._pcm8_cursor = 0
        self._last_ts_us = 0
        try:
            self._post("/face/close", {"session_id": self.session_id},
                       timeout=self._close_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.info("8PCM 重同步：会话释放失败（可忽略）：%s", exc)
        logger.warning("8PCM 时间轴不连续 —— 已重建人脸会话（dwell 清零）")

    def _probe_frame_size(self, jpeg: bytes) -> None:
        """首帧解析输入尺寸（UI 要把框坐标映射到显示区）。

        复用本地那份纯 Python SOF 解析，不引 cv2（OpenCV 在服务端）。
        """
        if self._frame_size is not None:
            return
        from .g1face_provider import jpeg_size
        sz = jpeg_size(jpeg)
        if sz is not None:
            self._frame_size = sz
            logger.info("远端人脸输入帧尺寸: %dx%d", sz[0], sz[1])

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

        if status in (409, 422) and self.pcm8_enabled:
            # 8PCM 门控下这两种都是**粘住**的（服务端记住期望值，之后每帧都拒）：
            #   422 = PCM 缺失 / 未对齐 / 不连续 / 含未来采样；
            #   409 = `PCM8 timestamp discontinuity: got X expected Y`
            #         —— 典型是「上一块服务端处理了但响应超时丢了」，重发同一块。
            # 唯一出路是 `/face/close` 重开会话（dwell 清零），否则人脸永久失效。
            self._resync_pcm8()
            return
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
