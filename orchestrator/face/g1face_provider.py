"""G1 人脸 provider —— **官方包的适配层**，只做「g1face dict → 我们的契约对象」翻译。

上游是 ``board-face-and-cloud-infer/G1/g1face``（见该仓库的 ``接口文档.md``）。
本文件**不自己绑 ctypes、不自己组装 state、不自己攒识别 burst、不自己维护
person_id 映射** —— 那些全部沿用官方实现：

    ctypes 绑定      g1face._abi.ensure_lib()      （含 ELF 架构自检 + ABI 版本门）
    每帧 state 组装   g1face.state.FrameStateBuilder + result_to_dict
    喂帧             g1face.util.feed_frame
    身份识别         g1face.runtime.G1IdentifyRuntime（自带后台线程 + FaceService）

**为什么之前自己写了一份**：上游的 Python 封装是 2026-09-17 之后才补齐的，
更早只有 C 头文件。现在整份换成官方的，别再抄一遍 —— 抄的那份已经落后了
（上游改了身份分档阈值、删了唤醒信号语义，我们的副本都不知道）。

与旧实现（``face/g1.py`` + ``face/local_provider.py``）的**语义差异**，都是有意为之：

  · **唤醒判据改用 ``dwell_ms``**，不再读 ``out.interacting``。上游已把
    ``interacting`` 标注为「保留字段但不对外」（``include/sdk_stream.h``），
    官方口径是调用方按 ``state["dwell_ms"]`` 自行判定。阈值 2000ms 与
    C 侧 ``wake_ms_high`` 和 IC 的 passerby 阈值同值。
  · **放弃自建的 PersonIdMap**。官方的 ``person_id`` 由 ``name_to_person_id()``
    正则解析 ``person_N`` 得到，而线上 ``face_db.npz`` 里存的是**真名**
    （崔雪涵 / 罗淇元 …）→ 解析全失败 → 所有人都落到同一个兜底值。
    所以它只当参考值透传，**UI 认人请用 ``uid``**。
  · **不再按 ``state_seq`` 去重**。官方明确：身份结果回来时 ``state_seq``
    可能不 +1，按 seq 去重会**恰好丢掉带身份的那一帧**。
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from .signals import FaceObservation, IdentityEvent

logger = logging.getLogger(__name__)

#: 本仓库 ``code/`` 层 —— G1 与 MiniCPM-o-Demo 同级
_DEFAULT_G1_ROOT = (Path(__file__).resolve().parents[3]
                    / "board-face-and-cloud-infer" / "G1")

#: ``platform.machine()`` → G1 的 ``lib/<arch>/`` 目录名
_ARCH_ALIASES = {"amd64": "x86_64", "x86_64": "x86_64",
                 "aarch64": "aarch64", "arm64": "aarch64"}

#: 官方身份状态机名 → 是否「在库里」。``recognized`` 命中已有条目，
#: ``enrolled`` 是在线注册的（我们默认关，见 ``no_enroll``）。
_IN_GALLERY = ("recognized", "enrolled")

#: 初始的「无身份」键，用来避免第一帧就发一条空身份事件。
_NO_IDENT = (None, "NONE", None, None)


def arch_name() -> str:
    """本机架构标签，也是 G1 ``lib/`` 下的目录名。"""
    import platform
    return _ARCH_ALIASES.get(platform.machine().lower(),
                             platform.machine().lower())


def resolve_g1_root(cfg=None) -> Tuple[Path, str]:
    """定位 G1 仓库根，返回 ``(路径, 来源说明)``。

    优先级：

      1. ``cfg.face_g1_root``（env ``ORCH_G1_ROOT``）—— 主旋钮
      2. ``<本仓库>/../board-face-and-cloud-infer/G1`` —— 现约定，自动探测
      3. ``dirname(cfg.face_model_dir)`` —— 向后兼容只设了 ORCH_FACE_MODELS 的老部署

    ⚠️ **``g1_root`` 与 ``model_dir`` 是两个不同的根，不能合并成一个变量**：
    检测模型（blazeface 等）由 C 侧按 ``g1_face_create(model_dir)`` 的入参解析，
    而身份模型 ``models/buffalo_l/`` 由 ``G1IdentifyRuntime`` 按 ``g1_root`` 解析
    （``g1face/runtime.py`` 的 ``_ensure_svc``）。
    """
    explicit = getattr(cfg, "face_g1_root", None) if cfg is not None else None
    if explicit:
        return Path(explicit), "ORCH_G1_ROOT"
    if (_DEFAULT_G1_ROOT / "g1face" / "__init__.py").is_file():
        return _DEFAULT_G1_ROOT, "自动探测（与本仓库同级的 board-face-and-cloud-infer/G1）"
    model_dir = getattr(cfg, "face_model_dir", None) if cfg is not None else None
    if model_dir:
        return Path(model_dir).resolve().parent, f"由 ORCH_FACE_MODELS 反推（{model_dir}）"
    return _DEFAULT_G1_ROOT, "默认路径（未探测到，多半需要设 ORCH_G1_ROOT）"


#: G1 仓库里可能同时躺着多份 x86_64 产物，**它们不通用**（实测）：
#:
#:   lib/<arch>/libsdk_stream.so  `build.sh` 的正规产物，ABI 3；
#:                                但它链接的 opencv 版本取决于**编译那台机器**——
#:                                在 opencv 4.10 的机器上编的，拿到只有 4.5 的
#:                                106 上会 `libopencv_imgcodecs.so.410 not found`
#:   src/libsdk_stream.so         编译中间产物，ABI 3、**不链 opencv**，
#:                                只差一个 libonnxruntime（旁边 lib/<arch>/ 就有）
#:   lib/libsdk_stream.so         旧约定路径的遗留，**没有 g1_face_abi_version**，
#:                                是 ABI 3 之前的构建 —— 能加载但 state 只出一次
#:
#: 所以**不能按目录名挑**（"按架构分目录 = 新的"这个直觉是错的），
#: 必须实测 + 验 ABI。下面的顺序是「先正规产物，再中间产物，最后遗留」。
_LIB_CANDIDATES = (
    "lib/{arch}/libsdk_stream.so",   # build.sh 正规产物
    "src/libsdk_stream.so",          # 编译中间产物（不链 opencv，最稳）
    "lib/libsdk_stream.so",          # 旧约定遗留（多半 ABI 太老）
)

#: 与 g1face 的 `ABI_VERSION` 对齐；`g1_face_abi_version()` 早于此就是旧构建。
_MIN_ABI = 3


def pick_g1_lib(g1_root: Path, explicit: Optional[str] = None) -> Tuple[Optional[str], str]:
    """挑一份**本机能真正加载、且 ABI 够新**的 ``libsdk_stream.so``。

    返回 ``(路径 | None, 说明)``。``explicit``（``ORCH_FACE_SO``）优先 ——
    但同样要过下面的校验，坏路径不会静默放行。

    ⚠️ **只看文件存不存在是不够的**：`lib/<arch>/` 和 `lib/` 两个目录下都
    可能有文件，一个缺 opencv、一个 ABI 太老。所以这里真的去
    ``ctypes.CDLL`` 一次并调 ``g1_face_abi_version()`` —— 多花几十毫秒，
    换掉一整类"服务起来了、人脸却是坏的"故障。
    """
    import ctypes

    tried = []
    if explicit:
        cands = [Path(explicit)]
    else:
        cands = [g1_root / t.format(arch=arch_name()) for t in _LIB_CANDIDATES]

    for path in cands:
        if not path.is_file():
            tried.append(f"{path.name}(不存在)")
            continue
        # 预载 onnxruntime（RTLD_GLOBAL）—— 让 soname 能解析到，
        # 这样**不需要**调用方设 LD_LIBRARY_PATH（部署"拉下来就能用"）。
        _preload_ort_for(path)
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            tried.append(f"{path}({str(exc)[:60]})")
            continue
        fn = getattr(lib, "g1_face_abi_version", None)
        if fn is None:
            # 能加载但太旧：用它跑不会报错，只会让 state 一辈子只出一次
            tried.append(f"{path}(没有 g1_face_abi_version，是旧构建)")
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = []
        ver = int(fn())
        if ver < _MIN_ABI:
            tried.append(f"{path}(ABI={ver} < {_MIN_ABI})")
            continue
        return str(path), f"ABI={ver}"
    return None, "；".join(tried) or "没有候选文件"


def _preload_ort_for(lib_path: Path) -> None:
    """把 libonnxruntime 以 ``RTLD_GLOBAL`` 预载，供 ``lib_path`` 解析 soname。

    ``src/libsdk_stream.so`` 的 RUNPATH 是 ``$ORIGIN/../../src/.ort_sdk/...``，
    而 ORT 实际在 ``G1/lib/<arch>/`` 下 —— 解析不到，于是
    ``libonnxruntime.so.1.16.3: cannot open shared object file``。
    普通 ``CDLL``（RTLD_LOCAL）**不够**，soname 解析要求全局符号可见。

    g1face 的 ``_preload_onnxruntime()`` 也会找这些目录，但它用的是
    RTLD_LOCAL、且**先命中哪个目录取决于顺序** —— 所以这里自己来一遍。
    """
    import ctypes

    roots = [lib_path.parent,                       # 与 .so 同目录
             lib_path.parent.parent / "x86_64",     # lib/<arch>/
             lib_path.parent.parent.parent / "lib" / arch_name(),
             lib_path.parent.parent / "lib" / arch_name(),
             lib_path.parent.parent / "src" / ".ort_sdk"]
    for d in roots:
        if not d.is_dir():
            continue
        cands = [d / "libonnxruntime.so.1.16.3", d / "libonnxruntime.so"]
        cands += sorted(d.glob("libonnxruntime.so.*"))
        for c in cands:
            if c.is_file():
                try:
                    ctypes.CDLL(str(c), mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    continue


def _ensure_g1face(g1_root: Path):
    """把 ``g1_root`` 塞进 ``sys.path`` 并 import ``g1face``。返回 ``(模块, 错误)``。

    ⚠️ **只能在函数内部调用** —— 本模块被 import 时 G1 可能根本不存在
    （开发机 win32 / 没同步该仓库），模块级 import 会让整个 orchestrator 起不来。
    """
    pkg = g1_root / "g1face"
    if not (pkg / "__init__.py").is_file():
        return None, f"g1face 包不存在：{pkg}"
    if str(g1_root) not in sys.path:
        sys.path.insert(0, str(g1_root))
    try:
        import g1face  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return g1face, None


#: JPEG 的 SOF 标记（帧头，含宽高）：C0-C3 / C5-C7 / C9-CB / CD-CF。
_SOF_MARKERS = frozenset(
    list(range(0xC0, 0xC4)) + list(range(0xC5, 0xC8))
    + list(range(0xC9, 0xCC)) + list(range(0xCD, 0xD0)))

#: 无载荷的独立标记
_STANDALONE = frozenset([0x01]) | frozenset(range(0xD0, 0xDA))


def jpeg_size(jpeg: bytes) -> Optional[Tuple[int, int]]:
    """纯 Python 读 JPEG 的 SOF 拿 ``(宽, 高)``；不是 JPEG / 读不出返回 None。

    **为什么不用 cv2**：这条路径在「关掉身份识别」时也要能用，而 cv2 只是
    insightface 的间接依赖（关识别时可能根本没装）。旧实现用 ``cv2.imdecode``
    且失败静默降级，结果是**永远拿不到 src_w/src_h**、前端框只能猜。

    为什么读原图尺寸就够：C 侧 ``g1_face_feed_mjpeg`` 是**按 JPEG 自身尺寸
    解码、不做 resize**（``src/sdk_stream.cpp``：``const int w = dec.cols, h = dec.rows;``），
    所以框坐标就在这张图的像素系里。
    """
    n = len(jpeg)
    if n < 4 or jpeg[0] != 0xFF or jpeg[1] != 0xD8:
        return None
    i = 2
    while i + 3 < n:
        if jpeg[i] != 0xFF:          # 没对齐标记，往前挪
            i += 1
            continue
        marker = jpeg[i + 1]
        if marker == 0xFF:           # 0xFF 填充
            i += 1
            continue
        if marker in _STANDALONE:
            i += 2
            continue
        if marker == 0xDA:           # SOS：后面是熵编码数据，不会再有 SOF
            return None
        seg_len = (jpeg[i + 2] << 8) | jpeg[i + 3]
        if seg_len < 2:
            return None
        if marker in _SOF_MARKERS:
            if i + 9 >= n:
                return None
            # SOF 载荷：精度(1B) 高(2B) 宽(2B) 分量数(1B) …
            h = (jpeg[i + 5] << 8) | jpeg[i + 6]
            w = (jpeg[i + 7] << 8) | jpeg[i + 8]
            return (w, h) if w > 0 and h > 0 else None
        i += 2 + seg_len
    return None


class G1FaceProvider:
    """``g1face`` 官方包的适配器。接口对齐 ``FaceWorker`` 的鸭子类型：

        ``process(jpeg, t) -> FaceObservation | None``
        ``poll_identity(t) -> IdentityEvent | None``
        ``close()``
        ``_frame_size``（供 UI 把框坐标映射到显示区）

    **单线程使用**（官方约定：``feed`` 单线程）—— 由 ``FaceWorker`` 的专用线程保证。
    """

    def __init__(self, g1_root: str, *,
                 model_dir: Optional[str] = None,
                 db_path: Optional[str] = None,
                 lib_path: Optional[str] = None,
                 identify: bool = True,
                 no_enroll: bool = True,
                 threshold: float = 0.36,
                 gpu_id: int = 0,
                 wake_dwell_ms: int = 2000,
                 warmup_iters: int = 2,
                 debug: bool = False) -> None:
        self.g1_root = Path(g1_root).resolve()
        self.model_dir = str(model_dir or (self.g1_root / "models"))
        self.db_path = db_path or None
        self.lib_path = lib_path or None
        self.wake_dwell_ms = int(wake_dwell_ms)
        self.debug = bool(debug)

        # ⚠️ 必须早于 create：G1 库**默认开 debug**，会往 CWD 写 ./debug/sess_*
        #    （视频最多约 4GB）。旧实现（face/g1.py）有同样的兜底，别丢。
        os.environ.setdefault("G1_FACE_DEBUG", "1" if self.debug else "0")

        # ⚠️ 这里**必须**把挑好（且已实测能加载）的那份写进 G1_LIB，而不是
        #    「设了就完事」：g1face 的 `_find_lib()` 是**先看 G1_LIB、再看
        #    lib/<arch>/**，而 `lib/<arch>/` 那份可能缺 opencv（见
        #    `_LIB_CANDIDATES` 的说明）。只把 ORCH_FACE_SO setdefault 进去、
        #    它指向别的文件时，g1face 仍会去挑 lib/<arch>/ 那份坏的。
        #    所以：**先选出可用的，再钉死给 g1face**。
        os.environ.pop("G1_LIB", None)
        picked, why = pick_g1_lib(self.g1_root, self.lib_path)
        if picked is None:
            raise RuntimeError(
                f"找不到可用的 libsdk_stream.so（{why}）。\n"
                f"  候选目录：{self.g1_root}/lib/<arch>/ 与 {self.g1_root}/src/。\n"
                f"  在**目标机器**上重编：cd G1 && bash build.sh（注意编译机的 "
                f"opencv 版本 —— 链接了 opencv 的那份拿到 opencv 更旧的机器上"
                f"会 not found）。")
        self.lib_path = picked
        os.environ["G1_LIB"] = picked
        logger.info("G1 库选定: %s（%s）", picked, why)

        mod, err = _ensure_g1face(self.g1_root)
        if mod is None:
            raise RuntimeError(
                f"g1face 不可用：{err}\n"
                f"  期望包路径 {self.g1_root / 'g1face'}。\n"
                f"  它来自与 MiniCPM-o-Demo 同级的 board-face-and-cloud-infer 仓库；"
                f"同步该仓库后，在**目标机器**上 cd G1 && bash build.sh 重编 .so。")
        self._mod = mod

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        # ensure_lib()：加载 .so + ELF 架构自检 + 绑全部签名 + 探 ABI 版本。
        # 失败直接抛 —— 比旧实现的 hasattr 兜底**更早更响**，这正是我们要的。
        lib = mod.ensure_lib()
        self.lib = lib
        logger.info("g1face 就绪: %s（ABI=%d sizeof(G1FaceResult)=%d 缺符号=%s）",
                    self._lib_path, mod.ABI_VERSION,
                    ctypes.sizeof(mod.G1FaceResult),
                    mod.MISSING_SYMBOLS or "无")

        self.handle = lib.g1_face_create(self.model_dir.encode("utf-8"))
        if not self.handle:
            raise RuntimeError(f"g1_face_create 失败（model_dir={self.model_dir}）")

        # ---- 身份识别：任一环节失败只降这一项，检测/唇动/唤醒照常 ----
        self.ident = None
        if identify:
            try:
                self.ident = mod.G1IdentifyRuntime(
                    lib, self.handle, g1_root=str(self.g1_root),
                    db_path=self.db_path if self.db_path else None,
                    threshold=float(threshold), no_enroll=bool(no_enroll),
                    gpu_id=int(gpu_id),
                    # ⚠️ 绝不落盘：上游默认允许在线注册 + maybe_save_db() 覆写
                    #    face_db.npz，在编排场景会把每个陌生人写进库。
                    save_db_path=None)
                self.ident.enable()
                if warmup_iters:
                    # 可以在主线程调 —— _Warmup 哨兵是在 g1-identify 线程里跑的
                    ms = self.ident.warmup(iters=int(warmup_iters), timeout=60.0)
                    if ms is not None:
                        logger.info("身份识别预热完成（%.0fms）", ms)
                logger.info("身份识别就绪：库=%s 阈值=%.2f 在线注册=%s",
                            self.db_path or "空库", float(threshold),
                            "关" if no_enroll else "开")
            except Exception as exc:  # noqa: BLE001
                logger.warning("身份识别不可用（%s: %s）—— 降级为仅检测/唇动/唤醒",
                               type(exc).__name__, exc)
                self.ident = None

        self.builder = mod.FrameStateBuilder()
        # 复用同一个结果结构体：省分配，也让收尾能拿到最后一次的 state
        self.out = mod.G1FaceResult()

        self._frame_size: Optional[Tuple[int, int]] = None
        self._last_state: Optional[dict] = None
        self._last_ident_key = _NO_IDENT
        self._pending_ident: Optional[IdentityEvent] = None
        self._cached_person_id = -1
        self._cur_t = 0
        # 诊断计数
        self.frames_seen = 0
        self.identify_calls = 0

    # ------------------------------------------------------------------ #
    #  只读属性
    # ------------------------------------------------------------------ #

    @property
    def _lib_path(self) -> str:
        try:
            return str(self._mod.LIB_PATH)
        except Exception:  # noqa: BLE001
            return "?"

    @property
    def available(self) -> bool:
        """身份识别是否可用（不可用时仍做检测/唇动/唤醒）。"""
        return self.ident is not None

    # ------------------------------------------------------------------ #
    #  每帧
    # ------------------------------------------------------------------ #

    def process(self, jpeg: bytes, t: int) -> Optional[FaceObservation]:
        """喂一帧 JPEG，返回观测；坏帧返回 ``None``。"""
        self._cur_t = t
        # 采集用**单调微秒**（接口文档要求），不是墙钟
        result = self._mod.feed_frame(
            self.handle, self.builder, self.ident,
            jpeg=jpeg, ts_us=int(time.monotonic() * 1e6), out=self.out)

        rc = result["rc"]
        if rc != 0:
            if rc == -2:
                logger.debug("JPEG 解码失败（%d 字节）", len(jpeg))
            else:
                logger.warning("g1_face_feed_mjpeg 返回 %d", rc)
            return None

        self.frames_seen += 1
        if self._frame_size is None:
            sz = jpeg_size(jpeg)
            if sz is not None:
                self._frame_size = sz
                logger.info("G1 输入帧尺寸: %dx%d", sz[0], sz[1])
            else:
                logger.warning("读不出 JPEG 尺寸（%d 字节）—— UI 的人脸框可能错位",
                               len(jpeg))

        # ---- state：**只在刷新帧非 None**（≈5Hz 心跳 + 离散变化 + 身份落地）----
        st = result["state"]
        if st is not None:
            self._last_state = st
            self._maybe_emit_identity(st)

        # ⚠️ dwell_ms / bbox_area_ratio 是**每帧实时值**，不随 state 快照推迟
        #    （C 侧专门修过「连续量不跟快照走」）。所以唤醒判据读 self.out，
        #    不是读上面那个 5Hz 的 st —— 否则唤醒最多要晚 200ms 才触发。
        raw = self.out.state
        dwell_ms = int(raw.dwell_ms)
        tid = int(raw.track_id)
        track_id = str(tid) if tid >= 0 else None

        f = result["face"]
        valid = bool(f["valid"])
        # 唤醒：track 连续在场够久。阈值与 C 侧 wake_ms_high / IC 的 passerby 同值。
        interacting = valid and dwell_ms >= self.wake_dwell_ms

        return FaceObservation(
            t=t,
            valid=valid,
            box=(float(f["left"]), float(f["top"]),
                 float(f["right"]), float(f["bottom"])) if valid else None,
            score=float(f["score"]),
            speaking=bool(result["speaking"]),
            lip_state=result["lip_state"] or "SILENT",
            interacting=interacting,
            # ⚠️ **每帧实时值**，不要从 `state` 里取 —— 那个只有 5Hz 刷新帧才有，
            #    在非刷新帧上取会得到 0（实测：唤醒恰好落在非刷新帧时，
            #    `begin` 事件的 dwell_ms 是 0，看起来像"没熬够就唤醒了"）。
            dwell_ms=dwell_ms,
            person_id=self._cached_person_id,
            track_id=track_id,
            identity_state=(st or {}).get("identity_state"),
            state=st,
            state_seq=int(st["state_seq"]) if st is not None else -1,
        )

    # ------------------------------------------------------------------ #
    #  身份
    # ------------------------------------------------------------------ #

    def _maybe_emit_identity(self, st: dict) -> None:
        """身份变化 → 攒一条待取事件。

        **从 state 里读身份，不重复调 ``after_feed()``** —— ``feed_frame`` 已在
        同一帧调过，而 ``after_feed`` 是从内部结果队列 **pop** 的，再调一次会
        打乱它的投递/回收节奏。

        身份落地那一帧必定刷新 state（``FrameStateBuilder.on_identify`` 会置
        ``_force``），所以「state 刷新时比对身份键」既不漏也不重复。
        """
        if self.ident is None:
            return
        # 顺带缓存 person_id（只作参考值，见模块 docstring）
        last = getattr(self.ident, "last", None)
        if last:
            pid = last.get("person_id_now") or last.get("person_id") or -1
            try:
                self._cached_person_id = int(pid)
            except (TypeError, ValueError):
                pass

        key = (st.get("identity_id"), st.get("identity_confidence"),
               st.get("display_name"), st.get("identity_state"))
        if key == self._last_ident_key:
            return                      # 与上次一样，不重复发
        self._last_ident_key = key

        uid = st.get("identity_id") or None
        istate = st.get("identity_state")
        sim = 0.0
        if last and last.get("uid") == uid:
            try:
                sim = float(last.get("similarity") or 0.0)
            except (TypeError, ValueError):
                sim = 0.0
        if uid:
            self.identify_calls += 1
            logger.info("身份识别：%s (uid=%s sim=%.3f state=%s)",
                        st.get("display_name") or "-", uid, sim, istate)
        self._pending_ident = IdentityEvent(
            t=self._cur_t,
            track_id=0,
            person_id=self._cached_person_id,
            uid=uid,
            # 身份不够高时官方就不给名字，这里原样透传（失败不留旧名字）
            name=st.get("display_name") or None,
            similarity=sim,
            is_enrolled=istate in _IN_GALLERY,
            identity_state=istate,
        )

    def poll_identity(self, t: int) -> Optional[IdentityEvent]:
        """取走待发身份事件（没有则 ``None``）。"""
        ev, self._pending_ident = self._pending_ident, None
        return ev

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """收尾：先等识别线程落地，再销毁句柄。

        ⚠️ ``G1IdentifyRuntime.close()`` 内部是**无超时的** ``worker.join()``
        （``g1face/runtime.py``）。一次卡住的 CUDA 推理会把整个会话收尾挂死，
        所以放短线程里带超时地等。
        """
        ident = self.ident
        self.ident = None
        joined = True
        if ident is not None:
            final = [None]

            def _drain() -> None:
                try:
                    final[0] = ident.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("ident.close 异常: %s", exc)

            th = threading.Thread(target=_drain, name="g1-ident-close", daemon=True)
            th.start()
            th.join(timeout=5.0)
            joined = not th.is_alive()
            if not joined:
                # 线程还活着 → 它随时可能回头调 g1_face_set_person_id(handle)。
                # 这时销毁句柄会段错误，宁可漏一个句柄也不能崩。
                logger.warning("身份识别线程未在 5s 内收干净 —— 跳过句柄销毁"
                               "（避免与之竞争导致崩溃）")
            elif final[0]:
                logger.info("身份识别收尾落地: state=%s name=%s",
                            final[0].get("state"), final[0].get("name") or "-")

        if joined and self.handle is not None:
            try:
                self.lib.g1_face_destroy(self.handle)
            except Exception as exc:  # noqa: BLE001
                logger.debug("g1_face_destroy 异常: %s", exc)
        self.handle = None
        logger.info("g1face 已关闭: 帧=%d 识别=%d", self.frames_seen,
                    self.identify_calls)
