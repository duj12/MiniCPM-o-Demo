"""InteractionCore + Agent 决策链路。

替代原来的「OmniLLM 直接生成回复」：

    InteractionCore（gRPC）  按感知状态做 P0 SOP 决策 → Action
    Agent Platform（HTTP）   按 Action 生成回复文本
    本包                    两者之间的胶水

组成：

  · :class:`InteractionClient` —— IC 的 gRPC 封装（写感知状态 / tick 取决策）
  · :class:`AgentClient`       —— Agent Platform 的 HTTP 客户端
  · :class:`InteractionDownstream` —— 实现 ``Downstream`` 协议，
    把 orchestrator 事件转成 IC 的 ``apply_*``，把 IC 的 Action 转成我们的动作

⚠️ **回复文本不经过本包** —— Agent 生成后调 orchestrator 的
``POST /v1/speak`` 接口，走既有的 ``Speak`` → TTS 链路。这样 Agent 保持
异步生成的自由（慢任务走 ``INSERT`` 的语义才成立）。
"""
from .agent import AgentClient
from .client import InteractionClient
from .downstream import InteractionDownstream

__all__ = ["AgentClient", "InteractionClient", "InteractionDownstream"]
