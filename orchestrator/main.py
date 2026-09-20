#!/usr/bin/env python3
"""Orchestrator 入口 —— 浏览器 WS 服务。

一条浏览器连接 = 一个 ``OrchestratorSession``，它把音视频输入扇出给
AEC/ASR/OmniLLM/人脸，并把下游决策变成 TTS 播报。

    python -m orchestrator.main --port 8100

环境变量见 ``orchestrator/config.py``。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Request, WebSocket  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from orchestrator.downstream.interface import Cancel  # noqa: E402
# ⚠️ 必须模块级导入：本文件有 ``from __future__ import annotations``，
# 所有注解都延迟成字符串，FastAPI 在**模块全局命名空间**里解析它们。
# 放在函数内 import 会导致 ``NameError: name 'WebSocket' is not defined``。

from orchestrator.config import Settings  # noqa: E402
from orchestrator.protocol import (  # noqa: E402
    AudioChunk,
    ErrorMsg,
    PlaybackReceiptMsg,
    SessionReady,
    decode_audio_b64,
    to_json,
)
from orchestrator.session import OrchestratorSession  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("orchestrator")


class SessionRegistry:
    """活跃会话登记（阶段 6 的观测/回收基础）。"""

    def __init__(self) -> None:
        self.sessions: dict = {}

    def add(self, sid: str, s: OrchestratorSession) -> None:
        self.sessions[sid] = s

    def remove(self, sid: str) -> None:
        self.sessions.pop(sid, None)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: dict(s.stats) for sid, s in self.sessions.items()
            },
        }


REGISTRY = SessionRegistry()

# 收尾任务集合：会话清理需要几秒（等 ASR 最终结果），而客户端断开时
# 处理协程已被取消 —— 清理必须放在独立任务里跑完，否则尾部数据丢失，
# 且 CancelledError 会逃逸成 uvicorn 的 ASGI 异常。
# 持有引用防止被 GC，done 回调里移除。
SHUTDOWN_TASKS: set = set()

# 声学延迟的持久化记录（按设备分组，跨会话复用）
from orchestrator.delay_store import DelayStore  # noqa: E402
DELAY_STORE = DelayStore()


async def _shutdown_session(sess: OrchestratorSession, sid: str,
                            cfg: Settings, tasks: list) -> None:
    """优雅收尾 + 关闭。在独立任务里跑，不受客户端取消影响。

    ⚠️ 后台任务（AEC/ASR 接收循环）**必须活到 drain 结束** —— drain 正是
    靠它们在收尾部的输出。所以取消放在最后，而不是在 handle_client 的
    finally 里。
    """
    try:
        await asyncio.wait_for(sess.drain("client_disconnect"),
                               timeout=cfg.drain_timeout_s)
    except asyncio.TimeoutError:
        logger.warning("[%s] 收尾超时（%.0fs），强制关闭", sid, cfg.drain_timeout_s)
    except asyncio.CancelledError:
        logger.info("[%s] 收尾被取消", sid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 收尾异常: %s", sid, exc)

    # drain 完成后才停后台任务
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    try:
        await sess.close("client_disconnect")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 关闭异常: %s", sid, exc)
    # 汇总到全局指标
    try:
        from orchestrator.metrics import GLOBAL
        GLOBAL.on_end(sess.metrics, failed=bool(sess.error))
    except Exception:  # noqa: BLE001
        pass
    logger.info("[%s] 会话结束\n%s", sid, sess.summary())


# ====================================================================== #
#  组件装配
# ====================================================================== #

async def build_session(sid: str, cfg: Settings, send_to_client,
                        hello: Optional[dict] = None) -> OrchestratorSession:
    """为一个浏览器连接装配所有组件。

    ``hello`` 是客户端首条 ``session.start`` 消息 —— **必须在装配前传入**，
    因为 AEC 模式要据此决定是否连云端（早先在装配后才读，导致用户选的
    "算法服务"被 config 默认的 "browser" 覆盖，云端 AEC 根本没连上）。
    """
    sess = OrchestratorSession(sid, config=vars(cfg))
    sess.send_to_client = send_to_client
    # 用户选择的 AEC 模式优先于 config 默认值
    if hello and hello.get("aec_mode"):
        sess.aec_mode = str(hello["aec_mode"])

    # ---- AEC ----
    # 只有「算法服务 AEC」模式才连云端；「浏览器原生 AEC」在前端做，
    # 连过去只是白跑一次云往返（实测该服务在此延迟下还相当于直通）。
    mode = getattr(sess, "aec_mode", cfg.aec_mode)
    sess.aec_mode = mode
    if cfg.enable_aec and mode == "service":
        from orchestrator.aec.client import AecClient
        sess.aec = AecClient(cfg.aec_url, connection_id=sid)
    elif mode == "browser":
        logger.info("[%s] AEC 模式=browser —— 不连云端 AEC（前端原生处理）", sid)

    # ---- ASR ----
    if cfg.enable_asr:
        from orchestrator.asr.client import AsrClient, AsrConfig
        # 客户端可在 session.start 里覆盖几个**调参性质**的字段
        # （vad_tail_sil / turnsense_incomplete_wait_ms / 两个置信度阈值），
        # 便于在页面上直接试不同切句敏感度，不用改代码重启。
        # 白名单见 AsrConfig.CLIENT_OVERRIDABLE —— 像 mode / chunk_size
        # 这种改了会让识别跑不起来的字段不开放。
        asr_cfg = AsrConfig()
        if hello:
            applied = asr_cfg.apply_overrides(hello.get("asr"))
            if applied:
                logger.info("[%s] ASR 参数被客户端覆盖: %s", sid, applied)
        sess.asr = AsrClient(cfg.asr_url, config=asr_cfg, wav_name=sid)

    # ---- OmniLLM ----
    if cfg.enable_omni:
        from orchestrator.omni.client import OmniClient
        from orchestrator.downstream.interface import (
            OmniDelta, OmniResponseDone, OmniTurnSense,
        )

        def on_omni_event(ev: dict) -> None:
            t = ev.get("type")
            now = sess.clock.now()
            if t == "turn.turnsense":
                sess.post_downstream(OmniTurnSense(
                    t=now, label=str(ev.get("label") or "")))
            elif t == "response.output.delta":
                sess.post_downstream(OmniDelta(
                    t=now,
                    delta_kind=ev.get("kind", "text"),
                    text=ev.get("text"),
                ))
            elif t == "response.done":
                # 计数器供 drain 观察「本轮是否已结束」——不去窥探事件队列
                # （run_downstream 是同一队列的消费者，会互抢）
                sess.stats["omni_done"] = sess.stats.get("omni_done", 0) + 1
                sess.post_downstream(OmniResponseDone(
                    t=now,
                    response_id=str(ev.get("response_id") or ""),
                    text=ev.get("text", "") or "",
                ))

        # ⚠️ 客户端传来的 system_prompt **优先于** config 默认值。
        #    早先这里只读 cfg，而 `hello["system_prompt"]` 那条路是个
        #    `pass` 空实现 —— 前端传了也没用。而且顺序本身也是错的：
        #    OmniClient 在这里创建，之后再想覆盖已经来不及了。
        omni_prompt = cfg.omni_system_prompt
        if hello and (hello.get("system_prompt") or "").strip():
            omni_prompt = str(hello["system_prompt"]).strip()
            logger.info("[%s] OmniLLM 系统提示词被客户端覆盖（%d 字）",
                        sid, len(omni_prompt))
        sess.omni = OmniClient(
            cfg.omni_url, system_prompt=omni_prompt,
            on_event=on_omni_event, verify_ssl=cfg.verify_ssl,
            turn_trigger=cfg.omni_turn_trigger,
        )

    # ---- TTS ----
    if cfg.enable_tts:
        if getattr(cfg, "mock_tts", False):
            from orchestrator.tts.client import MockTtsClient
            sess.tts = MockTtsClient()
        else:
            from orchestrator.tts.client import TtsClient
            sess.tts = TtsClient(cfg.tts_host, cfg.tts_port,
                                 tts_type=cfg.tts_type,
                                 speaker_id=cfg.tts_speaker_id)

    # ---- 参考轨 ----
    # ⚠️ 这里**不再**装配 AcousticDelayTracker。D 是每台设备离线测一次的
    # 固定常量（见 config.aec_default_delay_ms 与 tests/measure_delay.py），
    # 运行时自适应已证实会在真机上被噪声假峰钉死并永久失效。
    # 估计器本身还在（orchestrator/tools/delay_estimate.py），供离线工具用。
    from orchestrator.audio.ref_track import RefTrack
    sess.ref_track = RefTrack()

    # ---- downstream ----
    # 三种形态，优先级从高到低：
    #   ① InteractionCore + Agent（决策权在 IC）
    #   ② PassthroughDownstream（OmniLLM 回复 / ASR 直通；也是 ① 的降级目标）
    #   ③ None（mode=none，什么都不做）
    from orchestrator.downstream.passthrough import PassthroughDownstream
    mode = cfg.downstream_mode
    sess.downstream = None
    if mode != "none":
        # 流式合成只在 omni 模式 + TTS 客户端支持时生效（见 PassthroughDownstream）
        streaming = bool(getattr(cfg, "tts_streaming", False)) and \
            mode == "omni" and \
            bool(getattr(sess.tts, "supports_streaming", False))

        # IC 的地址：客户端可在 session.start 里覆盖（离线套件传 --ic-grpc）
        ic_cfg = (hello or {}).get("ic") or {}
        ic_target = str(ic_cfg.get("grpc") or cfg.ic_grpc)
        agent_target = str(ic_cfg.get("agent_url") or cfg.agent_url)
        # ⚠️ `echo` 模式保持纯桩 —— test_duplex_sim 靠它起一个"永不 Speak"
        #    的服务再自己替换 downstream，不能被 IC 抢走。
        # 客户端可显式覆盖（`ic.enabled`）—— replay 的 --no-ic 走这条。
        ic_on = ic_cfg.get("enabled")
        ic_on = bool(cfg.interaction_enabled) if ic_on is None else bool(ic_on)
        want_ic = ic_on and mode != "echo"

        if want_ic:
            from orchestrator.interaction import InteractionDownstream
            from orchestrator.protocol import IcDisplay

            def _on_ic_action(atype: str, sop, text) -> None:
                """IC 的决策 → UI（**只在非常见态回调**，不会刷屏）。"""
                sess._send_display(IcDisplay(
                    action=atype, sop=sop, text=text,
                    t_ms=int(sess.clock.seconds() * 1000),
                ))

            ic_ds = InteractionDownstream(
                ic_target, agent_target, session_id=sid,
                on_action=_on_ic_action)
            # ⚠️ 建连接**必须在这里**（不是 on_session_start）—— 连不上要
            #    立刻决定降级，而不是等会话跑起来才发现没有回复来源。
            if ic_ds.ic.connect():
                ic_ds.agent.start()
                sess.downstream = ic_ds
                sess.interaction = ic_ds
                logger.info("[%s] InteractionCore + Agent 已接管"
                            "（ic=%s agent=%s）", sid, ic_target, agent_target)
            else:
                # 降级：IC 不可用就回退 OmniLLM 回复。
                # ⚠️ 这条告警极其重要 —— 没有它，现象是"能识别、永远不回复"，
                #    与之前 OmniLLM 断线那个 bug 同一病理，极难排查。
                logger.warning(
                    "[%s] InteractionCore 不可用（%s：%s）"
                    "—— 降级为 OmniLLM 回复", sid, ic_target, ic_ds.ic.error)

        if sess.downstream is None:
            sess.downstream = PassthroughDownstream(
                mode=mode, streaming=streaming)
            if streaming:
                logger.info("TTS 流式合成已启用（LLM 文本 delta 边出边合成）")

    # ---- action 执行器 ----
    from orchestrator.actions.executor import ActionExecutor
    sess.executor = ActionExecutor(playback_delay_ms=cfg.playback_delay_ms)

    # ---- 人脸（阶段 4；默认关）----
    if cfg.enable_face:
        build_face(sess, cfg)

    return sess


def preflight_face(cfg: Settings) -> bool:
    """启动期人脸预检：只验证资产存在，**不加载模型**（避免空跑占 GPU）。

    为什么需要：人脸是会话建立时才懒加载的，如果路径配错，服务能正常
    起来、直到第一个用户连进来才失败 —— 那是很差的失败模式。这里在
    启动时就把问题暴露出来。
    """
    import os
    problems = []
    if not cfg.face_lib_path or not os.path.isfile(cfg.face_lib_path):
        problems.append(f"G1 库不存在: {cfg.face_lib_path}")
    if not cfg.face_model_dir or not os.path.isdir(cfg.face_model_dir):
        problems.append(f"模型目录不存在: {cfg.face_model_dir}")
    else:
        need = ["blazeface.onnx", "face_landmarks_op12.onnx", "anchors_192_v5.bin"]
        miss = [n for n in need
                if not os.path.isfile(os.path.join(cfg.face_model_dir, n))]
        if miss:
            problems.append(f"缺模型文件: {miss}")
    db_ok = bool(cfg.face_db_path) and os.path.isfile(cfg.face_db_path)
    if not db_ok:
        # 不阻断：没有离线库仍可做检测/唇动/唤醒，只是不做身份识别
        logger.warning("人脸：离线库不可用（%s）—— 将只做检测/唇动/唤醒，"
                       "不做身份识别", cfg.face_db_path)

    if problems:
        for p in problems:
            logger.error("人脸预检失败: %s", p)
        logger.error("人脸已开启但预检不通过 —— **会话将降级为无人脸**")
        return False
    logger.info(
        "人脸预检通过: lib=%s  models=%s  db=%s",
        os.path.basename(cfg.face_lib_path), cfg.face_model_dir,
        "可用" if db_ok else "不可用",
    )
    return True


def build_face(sess: OrchestratorSession, cfg: Settings) -> None:
    """装配人脸模块。任一前置缺失时**降级而非整体失败**。"""
    lib_path = cfg.face_lib_path
    model_dir = cfg.face_model_dir
    if not lib_path or not model_dir:
        logger.warning("人脸已开启但缺 face_lib_path/face_model_dir，跳过")
        return

    from orchestrator.face.local_provider import (
        LocalFaceProvider, PersonIdMap, load_face_service,
    )
    from orchestrator.face.worker import FaceWorker

    svc, err = (None, "未配置 face_db_path")
    if cfg.face_db_path:
        svc, err = load_face_service(model_dir, cfg.face_db_path)
    if svc is None:
        logger.warning("身份识别不可用（%s）—— 降级为仅检测/唇动", err)

    provider = LocalFaceProvider(
        lib_path, model_dir, face_service=svc,
        id_map=PersonIdMap(str(Path(cfg.face_db_path).with_suffix(".idmap.json"))
                           if cfg.face_db_path else None),
        identify_enabled=svc is not None,
    )
    sess.face_worker = FaceWorker(
        provider,
        on_wake=sess._face_cb("wake"),
        on_lip=sess._face_cb("lip"),
        on_identity=sess._face_cb("identity"),
        on_obs=sess._face_cb("obs"),      # 每帧观测，供 UI 叠加
    )
    logger.info("人脸模块已装配（身份识别=%s）", svc is not None)


# ====================================================================== #
#  WS 处理
# ====================================================================== #

async def handle_client(ws, cfg: Settings) -> None:
    """一条浏览器连接的完整生命周期。"""
    sid = uuid.uuid4().hex[:12]
    peer = getattr(ws, "remote_address", None)
    logger.info("[%s] 连接建立 peer=%s", sid, peer)
    t_connect = time.monotonic()

    # FastAPI 的 WebSocket 用 send_text/receive_text（不是 websockets 库的
    # send/recv）
    async def send_to_client(msg) -> None:
        await ws.send_text(to_json(msg))

    sess: Optional[OrchestratorSession] = None
    tasks: list = []
    try:
        # 第一条消息必须是 session.start
        first = await asyncio.wait_for(ws.receive_text(), timeout=15)
        hello = json.loads(first)
        if hello.get("type") != "session.start":
            await send_to_client(ErrorMsg(
                code="expect_session_start",
                message=f"首条消息必须是 session.start，收到 {hello.get('type')!r}",
            ))
            return

        sess = await build_session(sid, cfg, send_to_client, hello)
        identity = hello.get("identity", {}) or {}
        sess.config["identity"] = identity
        # 客户端能力位。有 ``playback_anchor`` 才会等浏览器的 armed 承诺
        # （精确落位）；没有则完全走原来的预测路径，零回归。
        sess.client_caps = set(hello.get("caps") or [])
        logger.info("[%s] 客户端能力: %s", sid,
                    sorted(sess.client_caps) or "（无 —— 参考轨将用预测落位）")
        # 前端可覆盖 AEC 模式（用户在 UI 上选）。
        # build_session 时还不知道用户的选择（hello 在这之后才读到），
        # 所以若模式从 browser 变成 service，这里补建 AEC 客户端。
        logger.info("[%s] session.start: aec_mode=%s 云端AEC=%s identity=%s",
                    sid, sess.aec_mode,
                    "已连" if sess.aec is not None else "未连",
                    sorted(identity.keys()))

        # 声学延迟初值：按设备标识查历史记录，冷启动即准
        # 优先用前端上报的设备键；没有就退化为"页面 + UA"的组合键
        client_key = str(
            identity.get("device_id")
            or identity.get("client_id")
            or hello.get("client_key")
            or "default"
        )
        seeded = int(hello.get("seeded_delay_samples") or 0)
        if seeded > 0 and sess.ref_track is not None:
            sess.ref_track.delay_samples = seeded
            sess.delay_source = "client"
            logger.info("[%s] 用前端上报的声学延迟 %d 采样", sid, seeded)
        else:
            sess.apply_delay_seed(client_key, DELAY_STORE)

        # system_prompt / asr 参数的客户端覆盖在 `build_session` 里就处理了
        # （必须在创建 OmniClient / AsrClient **之前**）—— 这里不再重复。

        # 连接外部服务
        if sess.aec is not None:
            await sess.aec.connect()
            # ASR 的时间戳原点 = 现在这一刻的会话采样位置
        if sess.asr is not None:
            await sess.asr.connect()
            sess.asr.stream_t0 = sess.clock.now()
        if sess.omni is not None:
            try:
                await sess.omni.connect()
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] OmniLLM 连接失败，降级为无 Omni: %s", sid, exc)
                sess.omni = None
        if sess.tts is not None:
            pass  # gRPC 懒连接

        await sess.start()
        REGISTRY.add(sid, sess)
        from orchestrator.metrics import GLOBAL
        GLOBAL.on_start()
        # 下发当前默认值供前端**预填输入框** —— 默认值只有一个真源
        # （config / 环境变量），客户端不自己写死。
        # ⚠️ 这里报的是**本会话实际生效**的值（含客户端刚覆盖的部分），
        #    这样"输入框里显示什么 = 真的在用什么"。
        _defaults: Dict[str, Any] = {
            "system_prompt": (sess.omni.system_prompt
                              if sess.omni is not None else ""),
        }
        if sess.asr is not None:
            from orchestrator.asr.client import AsrConfig as _AsrConfig
            _defaults["asr"] = {
                k: getattr(sess.asr.config, k)
                for k in _AsrConfig.CLIENT_OVERRIDABLE
            }
        await send_to_client(SessionReady(session_id=sid, defaults=_defaults))

        # ---- 后台任务 ----
        # 注意：不用 create_task(name=...)，那是 Python 3.8+ 的 API
        if sess.aec is not None:
            tasks.append(asyncio.create_task(sess.run_aec_recv()))
            tasks.append(asyncio.create_task(sess.run_fanout()))
        if sess.asr is not None:
            tasks.append(asyncio.create_task(sess.run_asr_recv()))
        if sess.omni is not None:
            tasks.append(asyncio.create_task(sess.omni.recv_loop()))
        tasks.append(asyncio.create_task(sess.run_downstream()))
        tasks.append(asyncio.create_task(sess.run_display()))
        tasks.append(asyncio.create_task(sess.run_tick(cfg.tick_interval_s)))
        if sess.face_worker is not None:
            tasks.append(asyncio.create_task(sess.run_face_signals()))

        # ---- 主接收循环 ----
        # 收消息用独立 task + wait：这样「等消息」与「等会话结束」可以
        # 同时进行，drain 在后台跑时循环仍能察觉 sess.closed 退出。
        # （直接给 receive_text 加 timeout 会取消它，FastAPI 的 WS 状态
        # 可能因此不一致。）
        recv_msg = asyncio.create_task(ws.receive_text())
        try:
            while not sess.closed:
                done, _ = await asyncio.wait(
                    {recv_msg}, timeout=0.3, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    continue
                try:
                    raw = recv_msg.result()
                except Exception:  # noqa: BLE001 — 连接关闭
                    break
                recv_msg = asyncio.create_task(ws.receive_text())
                await dispatch(sess, json.loads(raw))
        finally:
            recv_msg.cancel()

    except asyncio.TimeoutError:
        logger.warning("[%s] 等待 session.start 超时", sid)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] 会话异常: %s", sid, exc)
        if sess is not None:
            sess.error = str(exc)
    finally:
        if sess is not None:
            # ⚠️ 客户端断开时本协程会被**取消**，而收尾需要几秒（等 ASR
            # 最终结果）。直接在 finally 里 await 会立刻撞上 CancelledError，
            # 表现为 uvicorn 的 ASGI 异常 + 尾部数据丢失。
            #
            # 正确做法：把「收尾 + 关闭」注册为**独立的后台任务**（不属于
            # 本协程），并放在 SHUTDOWN_TASKS 里持有引用 —— 这样它不随本
            # 协程被取消，异常也不会逃逸到 uvicorn。
            t = asyncio.ensure_future(_shutdown_session(sess, sid, cfg, tasks))
            SHUTDOWN_TASKS.add(t)
            t.add_done_callback(SHUTDOWN_TASKS.discard)
            # 给它一小段时间跑完（客户端正常断开时通常足够）；
            # 超时就不等了，后台继续。
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout=0.05)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            logger.info("[%s] 客户端断开（存活 %.1fs），收尾转入后台",
                        sid, time.monotonic() - t_connect)
        # ⚠️ 后台任务的取消放在 _shutdown_session 里（drain 之后）——
        # drain 靠它们收尾部输出。这里不动它们。
        REGISTRY.remove(sid)


async def dispatch(sess: OrchestratorSession, msg: dict) -> None:
    """按消息类型分发。"""
    t = msg.get("type")
    if t == "audio":
        x = decode_audio_b64(msg.get("audio_base64", ""))
        # ctx_time/epoch 是浏览器 AudioContext 的锚点 —— 参考轨靠它做
        # 跨时钟域换算，缺了就只能退回预测落位（误差逐句变化）。
        await sess.on_audio(x, msg.get("t_ms", 0),
                            float(msg.get("ctx_time") or 0.0),
                            int(msg.get("epoch") or 0))
    elif t == "video_face":
        import base64
        raw = base64.b64decode(msg.get("frame_base64", ""))
        await sess.on_video_face(raw, msg.get("t_ms", 0))
    elif t == "video_omni":
        import base64
        raw = base64.b64decode(msg.get("frame_base64", ""))
        await sess.on_video_omni(raw, msg.get("t_ms", 0))
    elif t == "playback":
        # ⚠️ `sample_offset` 必须传下去 —— 它是前端报的「**实际播出**了
        # 多少采样」，是参考轨唯一可靠的观测值。早先这里漏传，服务端只能
        # 拿 `clock.now()` 猜，打断后新句往往已开始落位 → 猜错就切到新句。
        await sess.on_playback_receipt(
            msg.get("response_id", ""), msg.get("phase", "started"),
            float(msg.get("ctx_time", 0.0)), int(msg.get("seq", 0)),
            int(msg.get("sample_offset") or 0),
            start_ctx=float(msg.get("start_ctx") or 0.0),
            stop_ctx=float(msg.get("stop_ctx") or 0.0),
            epoch=int(msg.get("epoch") or 0),
        )
    elif t == "calibrate":
        await sess.start_calibration()
    elif t == "set_aec_mode":
        sess.set_aec_mode(str(msg.get("mode") or "browser"))
    elif t == "set_delay":
        sess.set_delay_ms(float(msg.get("delay_ms") or 0))
    elif t == "session.stop":
        # 客户端要求停止：先 drain 再 close（drain 会等 ASR 最终结果）。
        # 注意不能在 dispatch 里 await 太久 —— 主接收循环还等着收后续消息。
        # 所以 drain 在后台跑，主循环靠 sess.closed 退出。
        async def _stop():
            try:
                await asyncio.wait_for(sess.drain("client_stop"),
                                       timeout=20.0)
            except Exception:  # noqa: BLE001
                pass
            await sess.close("client_stop")
        sess._stop_task = asyncio.create_task(_stop())  # type: ignore[attr-defined]
    else:
        logger.debug("未知消息类型: %r", t)


# ====================================================================== #
#  FastAPI app
# ====================================================================== #

def create_app(cfg: Settings):
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Orchestrator", version="0.1.0")

    # 静态前端：复用 MiniCPM-o-Demo 的 static/ 目录（采集 worklet、
    # 播放器、duplex-utils 都在那里），Orchestrator 只需多提供自己的
    # 会话客户端与验证页。路径与本仓库既有约定一致（/static/...）。
    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index():
            from fastapi.responses import FileResponse
            page = static_dir / "orchestrator-test.html"
            if page.is_file():
                return FileResponse(str(page))
            return {"service": "orchestrator",
                    "hint": "static/orchestrator-test.html 不存在"}

    # ------------------------------------------------------------------ #
    #  Agent 回调：TTS 播报（InteractionCore + Agent 链路的**回程**）
    # ------------------------------------------------------------------ #
    #
    # PRD §2.8：TTS 是唯一开口通道，Agent 生成回复后直接把文本送进来。
    # 这两个接口**不产生任何新逻辑** —— 只是把文本转成既有的 `Speak`
    # action 投进下游队列，后面的 TTS / 参考轨 / 播放回执全部复用。
    #
    # ⚠️ **会话级接口**：必须指明是哪一路会话（`X-Session-Id` 头或
    #    `?session_id=`）。找不到就 404 —— 不能静默丢弃（Agent 会以为播了）。

    def _find_session(req) -> Optional[OrchestratorSession]:
        sid = (req.headers.get("x-session-id")
               or req.query_params.get("session_id") or "")
        if sid and sid in REGISTRY.sessions:
            return REGISTRY.sessions[sid]
        # 没指定就取唯一活跃会话（单会话部署的便利路径）
        if not sid and len(REGISTRY.sessions) == 1:
            return next(iter(REGISTRY.sessions.values()))
        return None

    def _queue_speak(sess: OrchestratorSession, act) -> None:
        """把 action 交给执行器。

        ⚠️ **不能投 ``sess._down_q``** —— 那个队列装的是"发给 downstream 的
        **事件**"，由 ``run_downstream`` 消费后调 ``downstream.on_event``。
        把 ``Speak``（一个 **action**）投进去，会被当成事件解析然后忽略，
        现象是接口返回 ``ok:true`` 但**永远不出声**（实测踩过）。

        正确路径是 ``execute_action`` —— 但它会 ``await``（走 TTS），
        所以**不能在这里直接 await**（会阻塞 HTTP 响应，且我们可能不在
        session 的 event loop 上下文里）。用 ``create_task`` 异步执行。
        """
        async def _do():
            try:
                await sess.execute_action(act)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Agent 播报执行失败: %s", sess.session_id, exc)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[%s] 无运行中的 event loop，播报被丢弃", sess.session_id)
            return
        sess.stats["agent_speaks"] = sess.stats.get("agent_speaks", 0) + 1
        t = loop.create_task(_do())
        # ⚠️ 记下 task 并挂回调 —— `create_task` 的异常**不会**自动冒出来
        #    （除非被 await）。早先没挂，`_do` 里若抛在 logger 之前/之外
        #    就完全静默，表现为"日志说接收了、但 TTS 一次没调"。
        t.add_done_callback(lambda fut: (
            logger.warning("[%s] 播报任务异常: %s", sess.session_id, fut.exception())
            if fut.exception() else None))
        logger.info("[%s] 播报任务已排入 event loop（loop=%s）",
                    sess.session_id, id(loop))

    @app.post("/v1/speak")
    async def speak(request: Request):
        """整段播报：Agent 把一条完整回复文本送进来。

        Agent 侧调用示例见 ``orchestrator/README.md`` 的「Agent 接入」一节。
        """
        body = await request.json()
        text = str(body.get("text") or "").strip()
        sess = _find_session(request)
        if sess is None:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "找不到活跃会话"
                         "（请在 X-Session-Id 头或 ?session_id= 里指明）"})
        if not text:
            return {"ok": False, "error": "text 为空，未播报"}
        # interrupt：先掐掉当前这句，再播新的
        if body.get("interrupt"):
            _queue_speak(sess, Cancel(reason="agent_speak"))
        from orchestrator.downstream.interface import Speak as SpeakAction
        _queue_speak(sess, SpeakAction(
            text=text,
            tts_type=str(body.get("tts_type") or "mltts"),
            speaker_id=body.get("speaker_id"),
        ))
        logger.info("[%s] Agent 播报（%d 字，interrupt=%s）: %s",
                    sess.session_id, len(text), bool(body.get("interrupt")),
                    text[:40])
        return {"ok": True, "session_id": sess.session_id}

    @app.post("/v1/speak/stream")
    async def speak_stream(request: Request):
        """流式播报：Agent 边生成边推片段（首声更早）。

        ⚠️ 同一轮的 ``stream_id`` 必须**唯一且稳定** —— 变了会被当成新的一轮
        （常量 stream_id 会导致第二轮被吞进上一轮的流，踩过这个坑）。
        最后一片传 ``is_final: true`` 收尾。
        """
        body = await request.json()
        sid = str(body.get("stream_id") or "").strip()
        sess = _find_session(request)
        if sess is None:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "找不到活跃会话"
                         "（请在 X-Session-Id 头或 ?session_id= 里指明）"})
        if not sid:
            return {"ok": False, "error": "stream_id 必填（同一轮内保持不变）"}
        from orchestrator.downstream.interface import Speak as SpeakAction
        text = str(body.get("text") or "")
        is_final = bool(body.get("is_final"))
        if text or is_final:
            _queue_speak(sess, SpeakAction(
                text=text, stream_id=sid, is_final=is_final,
                tts_type=str(body.get("tts_type") or "mltts"),
                speaker_id=body.get("speaker_id"),
            ))
        if is_final:
            logger.info("[%s] Agent 流式播报收尾 stream_id=%s",
                        sess.session_id, sid)
        return {"ok": True, "session_id": sess.session_id}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "active_sessions": len(REGISTRY.sessions)}

    @app.get("/stats")
    async def stats():
        return REGISTRY.stats()

    @app.get("/metrics")
    async def metrics_endpoint():
        """指标导出。生产可接 Prometheus 抓取，或供人工排查。"""
        from orchestrator.metrics import GLOBAL
        return {
            "global": GLOBAL.snapshot(),
            "active": REGISTRY.stats(),
        }

    # 注意：参数**不能**写字符串注解 —— 旧版 FastAPI（0.88）会把它当成
    # 查询参数去解析，握手直接 403。直接用真实类型。
    @app.websocket("/v1/orchestrator")
    async def orchestrator_ws(ws: WebSocket):
        await ws.accept()
        await handle_client(ws, cfg)

    # 独立的延迟校准通道 —— **不需要建立完整会话**。
    # 校准只要"播一段已知音频 + 收麦克风 + 算互相关"，不需要
    # ASR/OmniLLM/TTS，走完整会话既慢又强迫用户先"开始会话"。
    @app.websocket("/v1/calibrate")
    async def calibrate_ws(ws: WebSocket):
        await ws.accept()
        from orchestrator.calibrate_endpoint import handle_calibrate_ws
        await handle_calibrate_ws(ws, cfg)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrator WS 服务")
    p.add_argument("--host", default=Settings.host)
    p.add_argument("--port", type=int, default=Settings.port)
    p.add_argument("--downstream-mode", default=None,
                   choices=["asr", "omni", "echo", "none"])
    p.add_argument("--no-aec", action="store_true")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--no-omni", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--mock-tts", action="store_true",
                   help="用本地正弦代替 TTS 服务（离线验证链路）")
    p.add_argument("--log-level", default="info")
    # ---- HTTPS ----
    # ⚠️ 浏览器只在**安全上下文**里给 `getUserMedia`（麦克风/摄像头）：
    #    https:// 或 localhost。用 http:// + 局域网 IP 打开时，
    #    `navigator.mediaDevices` 直接是 undefined，页面根本采不到音视频。
    #    所以要用真设备采集就必须走 HTTPS。
    p.add_argument("--ssl-cert", default="",
                   help="HTTPS 证书（.pem）。与 --ssl-key 一起给才生效")
    p.add_argument("--ssl-key", default="",
                   help="HTTPS 私钥（.pem）")
    args = p.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    cfg = Settings.from_env()
    cfg.host, cfg.port = args.host, args.port
    if args.downstream_mode:
        cfg.downstream_mode = args.downstream_mode
    cfg.enable_aec = not args.no_aec
    cfg.enable_asr = not args.no_asr
    cfg.enable_omni = not args.no_omni
    cfg.enable_tts = not args.no_tts
    cfg.mock_tts = args.mock_tts  # type: ignore[attr-defined]
    if args.mock_tts:
        logger.info("使用 MockTtsClient（本地正弦，不连 TTS 服务）")

    # 启动期能力清单 —— 让"服务起来了"与"功能真的可用"区分开
    logger.info(
        "能力: aec=%s asr=%s omni=%s tts=%s face=%s downstream=%s",
        cfg.enable_aec, cfg.enable_asr, cfg.enable_omni, cfg.enable_tts,
        cfg.enable_face, cfg.downstream_mode,
    )
    if cfg.enable_face:
        preflight_face(cfg)

    app = create_app(cfg)

    import uvicorn
    ssl_kw = {}
    if args.ssl_cert or args.ssl_key:
        if not (args.ssl_cert and args.ssl_key):
            p.error("--ssl-cert 与 --ssl-key 必须成对给出")
        for path in (args.ssl_cert, args.ssl_key):
            if not os.path.isfile(path):
                p.error(f"证书文件不存在: {path}")
        ssl_kw = {"ssl_certfile": args.ssl_cert, "ssl_keyfile": args.ssl_key}
        logger.info("HTTPS 已启用：https://%s:%s（ws 自动升级为 wss）",
                    cfg.host, cfg.port)
    else:
        # ⚠️ 明确提醒：http + 非 localhost 时浏览器**不给麦克风/摄像头**，
        #    而这正是本服务最常见的用法（局域网里用另一台机器打开页面）。
        logger.warning(
            "未启用 HTTPS —— 从别的机器用 http://%s:%s 打开时，"
            "浏览器不会给麦克风/摄像头权限（需 --ssl-cert/--ssl-key）",
            cfg.host, cfg.port)

    uvicorn.run(app, host=cfg.host, port=cfg.port,
                log_level=args.log_level, **ssl_kw)


if __name__ == "__main__":
    main()
