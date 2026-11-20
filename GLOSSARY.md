# Interaction strategy (repair-list / IC)

Terms for judging repair-list demos and what InteractionCore treats as the speaking party.

## Language

**通过**:
A clip passes when replay against the live orchestrator shows the IC (and visible Agent) behavior required by that clip's SOP /「预期效果」/「预期结果」wording in `test_data/交互策略SOP视频测试用例.md` (v2, v1_new) or `test_data2/交互策略-算法修复清单.md`. Problem-description text is diagnosis context only.
_Avoid_: 听感过关, viz 主观过关, 根因已消除, 单测绿就算过

**行为契约**:
The shared pass checklist for clips that share the same「预期结果」wording (today: 75 and 09). Different problem descriptions do not create different pass rules.
_Avoid_: 按问题描述分套验收

**非主体语音**:
Speech that should not trigger ANSWER or a hard barge for the current interactive face. Covers off-screen voices, bystander talk, and other noise called out in demos.
_Avoid_: 画面外人声, 旁人打断, 杂音 (as separate policy kinds)

**主体语音**:
Speech attributed to the current interactive face, via open mouth now or mouth-backed for this transcript content key.
_Avoid_: 在场的人说的都算, 有 ASR 就有主体

**Clip**:
One SOP video entry in the TDD dataset: an id, suite, optional video path, expect wording, and observations.
_Avoid_: case, sample, demo row (when meaning the runnable unit)

**Suite**:
One of `v2`, `v1_new`, or `supplement` under `test_data/tdd/cases.json`.
_Avoid_: version, batch, SOP group (when meaning the TDD partition)

**Observation**:
A single machine-checkable pass condition for a Clip (events and/or pause/resume log evidence).
_Avoid_: check, assert, expect (when meaning the predicate itself)

**ANSWER 转写对齐**:
When IC emits ANSWER, the `speech.transcript` handed to Agent need only be **semantically / same-referent** with the clip's acceptance target (md / expect / observation needle). Exact wording match to the Demo script is not required.
_Avoid_: 要求转写与验收文字完全一致, ASR 原文必须等于 Demo 脚本

**TDD brief**:
The intent-and-ban document handed to an implementing agent (`test_data/tdd/BRIEF.md`), not the scorer or replay script.
_Avoid_: test plan dump, harness README

**话轮**:
ASR 判定的一句话单元。一个话轮的完成度只有 HIGH（大概率说完了）和 LOW（肯定没说完），没有 MEDIUM。
_Avoid_: 句子, 语音片段, utterance

**待回答话轮**:
一个已经到达 HIGH 的话轮，只差 StartLux 的结束判断尚未返回；回答所需的证据在到达 HIGH 时被冻结保存。
_Avoid_: 待答句, 挂起的半句, 缓存的话轮

**交接标记**:
IC 为把一句被放弃的语音交给 Agent 而停掉播报后欠下的一次交接；这句话处理完时应当清掉，不能留到下一句。
_Avoid_: 播报中断标记, YIELD 标记, handoff 标记
