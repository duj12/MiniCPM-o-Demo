# IC 用 StartLux 二分类判断停播、回答和结束

播报中用户说话时，IC 要分别回答三个问题：现在停不停播、这句话要不要交给 Agent 回答、这句话是不是结束会话。旧的 `utterance_role`（request/aside/exit/other）和 `exit_intent`（0–1 分）把这三个问题混在一起：`你不要再说了` 被 StartLux 判成 `exit` + `exit_intent=0.707`，IC 于是说“好的，再见”，而不是停播。代码用 `_STOP_PHRASES`、`_CLOSING_UTTERANCES` 等转写词表补偿，这类硬编码已被 `AGENTS.md` 禁止。

直接对 StartLux 的实验表明，三个信号可以干净地拆开：`barge_class`（停不停）、`expects_reply`（答不答）、`is_closing`（结束不结束）。正常问题 `expects_reply=yes 0.99`、叫停 `no 0.78`、结束 `is_closing=yes 0.82–1.00`、叫停 `is_closing=no 0.79–0.93`。

## Considered Options

- **保留 `utterance_role` + `exit_intent`，继续用转写词表补洞**：`exit` 同时表示“叫停”和“结束”，无法靠阈值区分；词表违反新的内容纪律，每换一种说法就要加词。
- **只用 `barge_class` + `addressed`**：`barge_class=true_interrupt` 对正常问题和叫停都成立，`addressed` 对两者也都高；无法决定答不答。
- **新增 `expects_reply` / `is_closing` 两个二分类问题（选中）**：每个问题只回答一件事，置信度直接对应一个动作；叫停和结束不再共享一个 `exit` 标签。

## Consequences

- 停播由播报中的 `barge_class=true_interrupt` 决定；HIGH 后 `expects_reply=yes` 才 ANSWER；`is_closing=yes` 才说结束语。
- `utterance_role` 与 `exit_intent` 退出决策和快照；`role.py` 的角色等待、`role=exit` 结束判断、`_STOP_PHRASES` / `_CLOSING_UTTERANCES` 一并删除。
- 低置信度偏向“不答、不结束”；原始转写变化时重问内容探针。
- HIGH 话轮还没有内容判单时冻成待回答话轮，最多等 `exit_verdict_wait_ms`（1.5s）；等待从探针真的问得出去（主体唇动对上）那一刻算起，否则探针刚回来就已经超时作废。超时仍没判单就不猜，不答（也不结束）。判单只认对冻住那句的票，不认中途一句附和的票。
- `expects_reply` 的问题措辞按「默认要回应」写：凡是针对数字人的实质发言（问句、请求、观点、反馈、夸赞、抱怨、自述、对交互的评价）都算 yes，只有纯附和、对旁人/环境的话、叫停和再见算 no。陈述句不是 no 的理由。
- StartLux 一次请求只接受特定组合：`expects_reply` + `is_closing` 可以同问，`addressed` + `affect` 可以同问；四个问题混在一起会 422。
- VLM 注意力词表（`_ATTENTION_AWAY_TOKENS`）暂时保留：StartLux 不做视觉判断，等 VLM 给出结构化字段后再迁移。
