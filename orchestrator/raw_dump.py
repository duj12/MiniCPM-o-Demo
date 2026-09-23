"""原始视频转储 —— 边收边写盘，会话结束时 mux 成**单个**文件。

## 为什么要这个

``orchestrator_replay.py`` 只吃**一个**文件（``--audio`` / ``--video``），
再从它派生音频和两路抽帧。它**不读** ``-face.mjpeg`` / ``-omni.mjpeg`` /
``.tsv`` 那些散装转储。所以要想离线复现，必须把「原始视频 + 麦克风音频」
合成一个容器文件。

## 与「送给算法的帧」的关系（**本模块存在的核心前提**）

前端只抓**一路**帧（1280×720 q≈0.9），**同一份字节**同时用于：
    - 人脸检测   （``video_face``）
    - OmniLLM    （``video_omni``，同一个 tick 复用同一份字节）
    - 落盘录制   （本模块）

所以录下来的视频**逐字节等于**服务端喂给算法的视频 —— 不存在「录制一个
格式、算法用另一个格式」的偏差。复现时算法看到的就是当时看到的。

## 为什么要边收边写（不能像 face/omni 那样攒在内存里）

1280×720 q0.9 约 133KB/帧 × 10fps ≈ 1.3MB/s。攒 5 分钟就是 ~400MB 驻留。
``_dump_face`` / ``_dump_omni`` 那套 ``list.append`` 的做法在小帧上没问题，
在这个量级上不行。所以这里：**收到就追加写盘**，内存里只留一份
``(t_ms, nbytes)`` 的索引（每帧 2 个整数）。

## 时间轴

``t_ms`` 用与 face/omni 转储**完全相同**的表达式（``clock.now()*1000//SR``），
即「会话采样轴毫秒」—— 而 ``-mic.wav`` 的第 0 个采样**就是** t=0。
两者同轴，这是音视频对齐的锚点。
"""

from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 会话时间轴的 tick —— 25fps，与前端抓帧定时器一致。
#: ⚠️ 必须与前端 `VIDEO_TICK_MS` 保持同一个值：帧按真实时间戳落盘（见
#: tsv），tick 只是 mux 时把帧铺回等间隔网格用的。tick 比实际抓帧密
#: 不会出错（重复上一帧），比实际稀则会**丢帧**。
RAW_TICK_MS = 40

#: 写线程队列上限。满了**丢新帧**（不是丢旧帧）——
#: 录制要的是"完整到最后一刻"，丢最新的会丢掉会话结尾。
_QUEUE_MAX = 64

#: mux 成功后是否保留 `-raw.mjpeg`。
#:
#: 默认**删**：它与 `-face.mjpeg` 逐字节相同（单路抓帧），单场就白占 740MB。
#: 设 `ORCH_KEEP_RAW_MJPEG=1` 可保留 —— 但只在你要拿它做逐帧取证时才有必要，
#: 正常情况下 `-face.mjpeg` 就是同一份字节。
_KEEP_MJPEG = os.environ.get("ORCH_KEEP_RAW_MJPEG", "0") == "1"


class RawVideoWriter:
    """把原始帧追加写进 ``<prefix>-<sid>-raw.mjpeg`` + ``.tsv``。

    线程模型对齐 ``orchestrator/face/worker.py``：专用线程 + 有界队列，
    **绝不阻塞事件循环**（``offer`` 只做 ``put_nowait``）。

    为什么不是攒在内存里最后一次性写 —— 见模块 docstring（内存）。
    为什么帧一落地就 flush —— 每次 10 个小写，代价可忽略；换来的是
    **进程被杀也留下可用的 tsv**。
    """

    def __init__(self, prefix: str, sid: str):
        base = Path(prefix)
        self.mjpeg_path = Path(f"{base}-{sid}-raw.mjpeg")
        self.tsv_path = Path(f"{base}-{sid}-raw.tsv")
        self.prefix = prefix
        self.sid = sid
        #: 每帧 (t_ms, nbytes) —— mux 时据此切 mjpeg，不需要解析 JPEG
        self.index: List[Tuple[int, int]] = []
        self._q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._fh = None
        self._tsv = None
        self.dropped = 0
        self.bytes_total = 0
        self._first_dims = ""

    def start(self) -> None:
        self.mjpeg_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.mjpeg_path, "wb")
        self._tsv = open(self.tsv_path, "w", encoding="utf-8")
        self._tsv.write("frame_index\tt_ms\tbytes\n")
        self._thread = threading.Thread(target=self._run, name="raw-dump",
                                        daemon=True)
        self._thread.start()
        logger.info("[%s] 原始视频转储开始：%s", self.sid, self.mjpeg_path)

    def offer(self, t_ms: int, jpeg: bytes) -> None:
        """入队一帧。**非阻塞** —— 队列满就丢这一帧并计数。

        ⚠️ 这里**不能**阻塞：调用方是 websocket 接收协程，堵住它等于
        把音频路径一起堵住（音频和视频在同一条接收循环里）。
        丢帧是可接受的 —— tsv 里留下时间空洞，mux 时用「保持上一帧」补上，
        对齐关系不破坏。
        """
        try:
            self._q.put_nowait((t_ms, jpeg))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        i = 0
        while True:
            item = self._q.get()
            if item is None:
                break
            t_ms, jpeg = item
            try:
                if not self._first_dims:
                    self._first_dims = _jpeg_dims(jpeg) or ""
                self._fh.write(jpeg)
                self._tsv.write(f"{i}\t{t_ms}\t{len(jpeg)}\n")
                self._fh.flush()
                self._tsv.flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 原始帧写入失败：%s", self.sid, exc)
                break
            self.index.append((t_ms, len(jpeg)))
            self.bytes_total += len(jpeg)
            i += 1

    def stop(self, timeout: float = 5.0) -> None:
        """收尾：排空队列、关文件。**幂等**。"""
        if self._thread is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            # 队列满时哨兵进不去 —— 直接标记线程结束即可（遗留帧无所谓，
            # 会话都结束了，重要的是别把收尾卡住）
            pass
        self._thread.join(timeout=timeout)
        for fh in (self._fh, self._tsv):
            try:
                if fh:
                    fh.close()
            except OSError:
                pass
        self._thread = None
        logger.info("[%s] 原始视频转储 %s：%d 帧 / %.1fMB（丢 %d 帧）首帧尺寸=%s",
                    self.sid, self.mjpeg_path, len(self.index),
                    self.bytes_total / 1e6, self.dropped,
                    self._first_dims or "?")


def _jpeg_dims(b: bytes) -> Optional[str]:
    """从 JPEG 头读出 ``WxH``（只看 SOF 段）。读不出返回 None。

    用途：**尺寸中途变化会让 ``-c:v copy`` 产出坏流**（mjpeg 里混着两种
    分辨率）。mux 前用它逐帧校验，不一致就放弃 mux 并明确告警，而不是
    悄悄产出一个播不了的文件。
    """
    i = 2
    n = len(b)
    while i < n - 9:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        # SOF0..SOF15（不含 DHT=0xC4 / JPG=0xC8 / DAC=0xCC）
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (b[i + 5] << 8) | b[i + 6]
            w = (b[i + 7] << 8) | b[i + 8]
            return f"{w}x{h}"
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = (b[i + 2] << 8) | b[i + 3]
        i += 2 + seg
    return None


def _wav_duration_ms(path: Path) -> int:
    """读 wav 时长（毫秒）。失败返回 0。"""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return int(w.getnframes() * 1000 / w.getframerate())
    except Exception:  # noqa: BLE001
        return 0


def _tick_indices(index: List[Tuple[int, int]], total_ms: int) -> List[int]:
    """把「按到达时间记录」的帧重采样到严格的 100ms tick 网格上。

    返回每个 tick 该用**第几帧**。帧在网格上「保持」到下一帧到来 ——
    于是：
      - 首帧之前的 ``[0, t_first)`` 自动保持第 0 帧（补上开场空隙）
      - 中途丢帧的空洞自动用上一帧填上，**不产生时间轴错位**
      - 每帧都仍是**收到过的原始字节**，没有任何重编码

    ⚠️ 为什么不用 ``-f concat``：实测它会把每段的 duration **量化到
    40ms 网格**（0.1s → 0.12s），而 ``-c:v copy`` + ``-framerate`` 的
    时长是精确的（3000 帧零漂移）。
    """
    if not index:
        return []
    end = max(total_ms, index[-1][0] + RAW_TICK_MS)
    n_ticks = int(math.ceil(end / RAW_TICK_MS))
    out: List[int] = []
    i = 0
    for k in range(n_ticks):
        t = k * RAW_TICK_MS
        while i + 1 < len(index) and index[i + 1][0] <= t:
            i += 1
        out.append(i)
    return out


def mux_raw_video(prefix: str, sid: str, *,
                  timeout_s: float = 180.0) -> Optional[str]:
    """把 ``-raw.mjpeg`` + ``-mic.wav`` 合成 ``-raw.mkv``。

    **视频 ``-c:v copy``**：帧字节与服务端收到的**一模一样**，绝不重编码
    （与 ``flush_video_dump`` 的同一条理由 —— 验证要用原始字节）。
    **音频 ``-c:a copy``**：mic.wav 原样进容器，样本逐位不变。

    ⚠️ 容器必须是 **matroska(.mkv)**：mp4 **装不下 pcm_s16le**
    （``codec not currently supported in container``）。mkv 也已经在
    ``orchestrator_replay.py`` 的视频扩展名列表里，直接可喂。

    ⚠️ **不加 ``-shortest``**：它会在视频比音频短时**截断麦克风音频** ——
    而 replay 正是拿这段音频当"麦克风输入"的。
    ⚠️ **不加 ``-t``**：实测按块截断会引入尾部补零（32768 vs 32000 采样），
    完整拷贝则是逐样本精确的。

    返回生成的文件路径；跳过或失败返回 None（**永不抛异常**给调用方，
    收尾流程不该被它拖垮）。
    """
    mjpeg = Path(f"{prefix}-{sid}-raw.mjpeg")
    tsv = Path(f"{prefix}-{sid}-raw.tsv")
    mic = Path(f"{prefix}-{sid}-mic.wav")
    out = Path(f"{prefix}-{sid}-raw.mkv")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("[%s] 未找到 ffmpeg —— 跳过 mux（原始文件保留：%s）",
                       sid, mjpeg)
        return None
    if not mjpeg.is_file() or not tsv.is_file():
        return None          # 没录到帧：静默（无摄像头的会话就是这条路径）
    if not mic.is_file():
        logger.warning("[%s] 没有 mic.wav（未开音频转储？）—— 跳过 mux", sid)
        return None

    # 读索引（与写线程写出的 tsv 同源，但直接用文件保证与 mjpeg 一致）
    index: List[Tuple[int, int]] = []
    with open(tsv, "r", encoding="utf-8") as f:
        next(f, None)                                    # 表头
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                index.append((int(p[1]), int(p[2])))
    if len(index) < 2:
        logger.warning("[%s] 原始视频帧太少（%d 帧）—— 跳过 mux",
                       sid, len(index))
        return None

    # 逐帧校验尺寸一致（混尺寸会让 -c:v copy 产出坏流）
    dims = None
    with open(mjpeg, "rb") as fh:
        for _, n in index:
            b = fh.read(n)
            if len(b) != n:
                logger.warning("[%s] mjpeg 长度与 tsv 不符 —— 跳过 mux", sid)
                return None
            d = _jpeg_dims(b)
            if dims is None:
                dims = d
            elif d != dims:
                logger.warning("[%s] ⚠️ 原始帧尺寸中途变化（%s → %s）—— "
                               "跳过 mux，避免产出坏流。原始文件保留：%s",
                               sid, dims, d, mjpeg)
                return None

    audio_ms = _wav_duration_ms(mic)
    ticks = _tick_indices(index, audio_ms)
    if not ticks:
        return None

    fps = 1000.0 / RAW_TICK_MS
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-thread_queue_size", "512",
           "-f", "image2pipe", "-framerate", (f"{fps:g}"),
           "-vcodec", "mjpeg", "-i", "pipe:0",
           "-i", str(mic),
           "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "copy",
           "-f", "matroska", str(out)]
    logger.info("[%s] 正在合成复现文件：%d 帧 / 音频 %.1fs → %s",
                sid, len(ticks), audio_ms / 1000.0, out.name)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = b""
    try:
        # 按 tick 顺序流式喂帧 —— 内存里只留当前这一帧
        prev = -1
        cur = None
        with open(mjpeg, "rb") as fh:
            for k in ticks:
                while prev < k:
                    prev += 1
                    cur = fh.read(index[prev][1])
                    if len(cur) != index[prev][1]:
                        raise ValueError("mjpeg 提前结束")
                proc.stdin.write(cur)
        # ⚠️ **必须用 `close()` + `wait()`，不能再调 `communicate()`**。
        #    `communicate()` 内部会再 flush 一次 stdin，而此时 stdin 已经
        #    关了 → `ValueError: flush of closed file`。实测踩过：ffmpeg
        #    其实**正常跑完了**，但这个异常让调用方以为失败，mkv 虽然写出来
        #    了却被当成没产出（异常还被 except 吞掉，只留一行 warning）。
        proc.stdin.close()
        proc.stdin = None
        err = proc.stderr.read() if proc.stderr else b""
        proc.wait(timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError((err or b"").decode("utf-8", "replace")[-500:])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 合成失败：%s", sid, exc)
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return None
    # 成功判据用**文件真的在且非空**，不只信 returncode —— 多一道保险，
    # 免得再出现"其实写出来了却报失败"的反向情况
    if not out.is_file() or out.stat().st_size == 0:
        logger.warning("[%s] 合成后文件不存在或为空：%s", sid, out)
        return None

    # ⚠️ **mjpeg 是 mkv 的逐字节前置**，而且与 `-face.mjpeg` **完全相同**
    #    （前端单路抓帧，同一份字节既喂人脸又落盘 —— 见本模块 docstring）。
    #    也就是说这 740MB 存了**两份**、且内容还能从 face.mjpeg 复原。
    #    所以合成成功后删掉它，只留 mkv + tsv。
    #    `-face.mjpeg` **保留**（它是"算法实际看到什么"的原始证据，
    #    验证要用原始字节，不能拿 mkv 代替）。
    if not _KEEP_MJPEG:
        try:
            mjpeg.unlink()
            logger.info("[%s] 已删除与 face.mjpeg 重复的 -raw.mjpeg"
                        "（省 %.0fMB；face.mjpeg 是同一份字节，仍在）",
                        sid, mjpeg.stat().st_size / 1e6
                        if mjpeg.exists() else 0)
        except OSError as exc:
            logger.warning("[%s] 删除 -raw.mjpeg 失败：%s", sid, exc)
    logger.info("[%s] ✅ 复现文件已生成：%s（%.1fMB，video=copy audio=copy，"
                "尺寸=%s，%d 帧）。复现：python orchestrator_replay.py "
                "--video %s --face-fps %d --omni-fps 1",
                sid, out, out.stat().st_size / 1e6, dims, len(ticks), out,
                int(round(1000.0 / RAW_TICK_MS)))
    return str(out)
