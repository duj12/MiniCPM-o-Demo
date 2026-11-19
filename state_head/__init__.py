"""带置信度的 2.1-lite state head 服务（OneJev / Flide ``/v1/systemone``）。

这个包与 ``orchestrator/`` **互不 import** —— 它是旁挂容器里的独立进程，
通过 HTTP 被编排调用。两边的契约靠：

* 语义：``orchestrator/docs/Social_State_Schema_v2_1_Lite.md``
* 载荷：``state_head/assemble.py`` 产出的 E2 JSON
* 测试：``orchestrator/tests/test_state_merge.py`` 只读同一份枚举

模块分工见 ``state_head/README.md``。
"""

__all__ = ["questions", "assemble", "backends", "scheduler", "app"]
