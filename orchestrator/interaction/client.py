"""Local InteractionCore method/signature and Agent state normalization helpers."""
from __future__ import annotations
import inspect
from typing import Any

def _signature_accepts(fn: Any, keys: tuple) -> bool:
    """``fn`` 是否接受**全部** ``keys``（按关键字）—— IC 版本探测的唯一手段。

    P05 给 ``apply_asr`` 加了感知字段、新开了 ``apply_vision``；之前的 IC
    两个都没有。硬发过去不会"报错"，而是**每秒几十条 TypeError**：gRPC 模式
    在消费线程里被吞成 warning（见 ``_run``），进程内模式只累加
    ``failures`` 计数 —— 调用侧完全看不出来。所以调用前先问一句。

    签名是 ``*args, **kwargs`` 的一律当作"接受"：``Engine`` 的 ``_locked``
    壳子就是这种形态，**不能**拿它做判断（进程内模式探测的是
    ``engine.state`` 上的真方法，见 ``InProcessICClient.supports``）。
    """
    if fn is None:
        return False
    if not keys:
        return True
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(k in params for k in keys)


def normalize_agent_status(raw):
    """把大小写不限的 status 字串归一到 `AgentStatus` 枚举；非法返回 None。

    ⚠️ **必须映射到枚举，不能只把字符串大写。** IC 里两个枚举的大小写约定
    是**相反**的（实测）：

        Confidence   值是大写   NONE / LOW / MEDIUM / HIGH
        AgentStatus  值是**小写** idle / busy / pending_announce

    而 `policy.py` 是拿 `== AgentStatus.PENDING_ANNOUNCE` 判的。往状态里存
    一个大写字串 `'PENDING_ANNOUNCE'`，比较就是 **False**
    （`'PENDING_ANNOUNCE' != 'pending_announce'`）—— 不报错、不告警，
    **SOP 39 只是永远不生效**。另一个方向更直接：`AgentStatus('BUSY')`
    会抛 `ValueError`，把整次写入打掉。

    所以统一走枚举反查：大小写两种写法都能对，且写进去的一定是枚举成员。
    """
    try:
        from interaction.state import AgentStatus
    except ImportError:      # interaction 包没装 —— 调用方会降级
        return None
    key = str(raw).strip().upper()
    for member in AgentStatus:
        if member.name.upper() == key:
            return member
    return None


def agent_patch_kwargs(status=None, session_end_pending=None) -> dict:
    """把 `apply_agent` 的两个入参归一成 `Engine.apply_agent` 的 kwargs。

    语义与 proto 的 `ApplyAgent` 逐字对齐：**省略（None）= 不改该字段**。
    非法 status 归一为「不改」而不是抛 —— 一个坏字段不该把整次写入打掉。
    """
    import logging

    kwargs = {}
    if status:
        norm = normalize_agent_status(status)
        if norm is None:
            logging.getLogger(__name__).warning(
                "apply_agent 收到未知 status=%r —— 已忽略该字段"
                "（合法值 IDLE/BUSY/PENDING_ANNOUNCE，大小写不限）", status)
        else:
            kwargs["status"] = norm
    if session_end_pending is not None:
        kwargs["session_end_pending"] = bool(session_end_pending)
    return kwargs
