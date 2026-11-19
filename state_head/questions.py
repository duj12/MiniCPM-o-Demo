"""题集 —— **唯一真值源**：`Social_State_Schema_v2_1_Lite.md` 的槽位 → typed question。

一次前向回答全部问题（Flide 的 ``/v1/systemone`` 语义），所以这里的每一条
都必须满足三件事：

1. **id 就是 E2 路径**（``attn.orientation`` / ``beh.motion`` …）—— 组装侧
   （``assemble.py``）不做任何名字映射，问句改了 id 就是协议变更；
2. **枚举值一律英文 token**（schema §4.5）—— 中文只出现在 ``instructions``
   与 ``criteria`` 的**描述**里，不进答案；
3. **选项越少越好** —— 3090 上与 Qwen3-Omni 共卡，题数与选项数直接等于
   延迟。见 ``build_questions`` 的 ``profile``。

刻意**不发**的槽位（留给 Qwen3-Omni 的自由文本，或由编排侧的空值语义表达）：

* ``FEATURE.appearance`` / ``FEATURE.accessories`` —— 自由文本；
* ``ENV.objects`` / ``POSITION.relative_positions`` / ``description`` —— 自由文本；
* ``ACTION.events[].type`` 里 ``other`` 的描述 —— schema 要求写进 description。

``score`` 型问题返回的是**期望档位**（0..n-1 的浮点），不是概率向量，
所以 ``count`` / ``duration_s`` 这类数值槽位用它；``choice`` 才有逐选项概率。
"""
from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
#  枚举白名单（与 docs/Social_State_Schema_v2_1_Lite.md §3 逐字一致）
# --------------------------------------------------------------------------- #

ATTENTION_ORIENTATION: Dict[str, str] = {
    "facing_camera": "面向镜头（正对）",
    "angled": "斜对镜头",
    "sideways": "侧身",
    "back_to_camera": "背对镜头",
    "unknown": "无法判断",
}

ATTENTION_GAZE: Dict[str, str] = {
    "at_camera": "注视镜头",
    "at_screen": "注视屏幕",
    "at_person": "看着某人",
    "at_object": "看着某物",
    "wandering": "视线游移",
    "unknown": "无法判断",
}

#: 短时动作（``noul``，每题一个概率）。
#:
#: ⚠️ 顺序即优先级：多个动作同时过阈值时，按这里的顺序取前 ``ACTION_MAX_EVENTS``
#: 个。``nod`` / ``shake_head`` 语义互斥，放最前面保证它们先占位。
ACTION_EVENTS: Dict[str, str] = {
    "nod": "点头",
    "shake_head": "摇头",
    "wave": "挥手",
    "thumbs_up": "竖大拇指",
    "point": "指向",
    "raise_hand": "举手",
    "clap": "鼓掌",
    "hand_over_object": "递交物品",
}

#: ``STATE_TIER2=1`` 时才问的动作（9 类里的长尾）—— 每题都是一次前向增量，
#: 默认关掉是为了省延迟，不是因为语义不同。
ACTION_EVENTS_TIER2: tuple = ("raise_hand", "clap", "hand_over_object")

#: 短时动作的观测窗口。写进 instructions：没有这个锚，模型会把"上一分钟
#: 做过的手势"当成当前动作（SOP07 的失败模式）。
ACTION_WINDOW_HINT = "最近 1 到 2 秒内"

EMOTION_LABELS: Dict[str, str] = {
    "happy": "高兴（明显微笑、外放愉快）",
    "satisfied": "满意（放松 + 认可，脸颊放松、嘴角轻扬）",
    "neutral": "平静（中性表情）",
    "confused": "困惑（眉间收紧、眉头内侧上挑）",
    "dissatisfied": "不满（轻中度负面，无明显攻击性）",
    "impatient": "不耐烦（明显烦躁、紧锁眉头、目光锐利）",
    "angry": "愤怒（强烈愤怒线索、嘴角下压）",
    "sad": "悲伤（嘴角下垂、眉部呈悲伤特征）",
    "surprised": "惊讶（睁大眼、张嘴）",
    "other": "有其他明显情绪，但不属于上述类别",
    "unknown": "证据不足，或人脸不可见",
}

POSTURE: Dict[str, str] = {
    "standing": "站立",
    "sitting": "坐",
    "squatting": "蹲",
    "bending": "弯腰/俯身",
    "lying": "躺",
    "unknown": "无法判断",
}

MOTION: Dict[str, str] = {
    "stationary": "静止/保持不动",
    "walking": "行走",
    "approaching": "靠近",
    "moving_away": "远离",
    "turning": "转身",
    "moving_sideways": "横向移动",
    "unknown": "无法判断",
}

ACTIVITY: Dict[str, str] = {
    "using_phone": "使用手机",
    "reading_or_viewing": "阅读/观看",
    "working": "工作",
    "observing_object": "观察展品或物体",
    "other": "其他持续活动",
    "unknown": "无法判断",
}

PERSON_COUNT: Dict[str, str] = {
    "0": "画面里没有人物",
    "1": "画面里有 1 个人",
    "2": "画面里有 2 个人",
    "3+": "画面里有 3 个或更多人",
}

INTERACTION: Dict[str, str] = {
    "none": "人物之间没有可观察的互动",
    "talking": "人物之间在交谈",
    "facing_each_other": "人物相互朝向（面对面），不一定是交谈",
    "cooperating": "人物在协作",
    "other": "有其他可观察互动",
    "unknown": "无法判断",
}

REL_TARGET: Dict[str, str] = {
    "another_person": "另一人",
    "camera_or_device": "镜头或设备",
    "object": "物体",
    "none": "没有交互对象",
    "unknown": "无法判断",
}

SCENE: Dict[str, str] = {
    "home": "家居",
    "office": "办公室",
    "meeting_room": "会议室",
    "store": "商店/门店",
    "exhibition": "展厅",
    "corridor": "走廊",
    "street": "街道",
    "public_space": "公共空间/广场",
    "nature": "自然户外",
    "vehicle": "车内",
    "other": "其他场景",
    "unknown": "无法判断",
}

LIGHTING: Dict[str, str] = {
    "bright": "明亮",
    "normal": "正常",
    "dim": "偏暗",
    "backlit": "逆光",
    "unknown": "无法判断",
}

POSITION_REGION: Dict[str, str] = {
    "top_left": "左上",
    "top_center": "上中",
    "top_right": "右上",
    "middle_left": "左中",
    "center": "中央",
    "middle_right": "右中",
    "bottom_left": "左下",
    "bottom_center": "下中",
    "bottom_right": "右下",
    "unknown": "无法判断",
}

#: ``act.count`` 的 ``score`` 档位 —— 索引即期望档位，``assemble`` 按
#: ``1 + round(expected)`` 折算成整数次数并夹到 ``[1, 4]``。
ACTION_COUNT_LEVELS: List[str] = ["1 次", "2 次", "3 次", "4 次及以上"]

#: ``beh.duration`` 的 ``score`` 档位 —— 折算表见 ``assemble.DURATION_SECONDS``。
BEHAVIOR_DURATION_LEVELS: List[str] = [
    "不到 2 秒",
    "2 到 5 秒",
    "6 到 10 秒",
    "超过 10 秒",
]


# --------------------------------------------------------------------------- #
#  题集装配
# --------------------------------------------------------------------------- #

def _choice(instructions: str, criteria: Dict[str, str]) -> Dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def _noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _score(instructions: str, levels: List[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def build_questions(stage: str = "turn", tier2: bool = False,
                    include_lighting: bool = True,
                    include_scene: bool = None) -> Dict[str, Dict[str, Any]]:
    """组装本轮题集。

    ``stage="turn"`` = 19 问（不含 ``env.scene``），``"full"`` = 20 问。
    题数是延迟的主旋钮，全部由 env 控制、不改代码：

    * ``include_scene`` —— 默认 ``stage == "full"``（场景类型是静态的，
      起点问一次足够；schema §5.2 也允许稳定槽位在 E2 里省略）；
    * ``include_lighting`` —— 默认开；
    * ``tier2`` —— 追加 3 个长尾动作。

    返回**有序** dict（``dict`` 自 3.7 起保序），组装侧按同一顺序遍历。
    """
    q: Dict[str, Dict[str, Any]] = {}

    # ---- 0. 门控：画面里有没有人 -------------------------------------- #
    #
    # 必须第一问：它决定 ACTION.events 是 ``[]`` 还是 ``null``（schema §5.1
    # —— 这两件事对策略完全相反），也决定枚举槽位是 ``unknown`` 还是 ``null``。
    q["person_visible"] = _noul(
        "画面里能看到一个清晰的人物主体（人可以被部分遮挡，但主体可见）")

    # ---- 1. ATTENTION ------------------------------------------------- #
    q["attn.orientation"] = _choice(
        "主要人物的身体朝向镜头是哪种？合并身体与头部的朝向，取整体可见朝向。",
        ATTENTION_ORIENTATION)
    q["attn.gaze"] = _choice(
        "主要人物的视线落在哪里？朝向正对镜头不代表在注视镜头。",
        ATTENTION_GAZE)

    # ---- 2. ACTION（每类一个 noul，次数用一个 score）------------------ #
    active_events = {k: v for k, v in ACTION_EVENTS.items()
                     if tier2 or k not in ACTION_EVENTS_TIER2}
    for name, zh in active_events.items():
        q[f"act.{name}"] = _noul(
            f"{ACTION_WINDOW_HINT}，画面里的人物做了「{zh}」这个动作"
            f"（即使动作在生成时已经结束，只要落在观测窗口内就算）")
    q["act.count"] = _score(
        "刚才出现的主要动作一共重复了几次？",
        ACTION_COUNT_LEVELS)

    # ---- 3. EMOTION（**纯视觉**；语音侧在编排侧晚融合）---------------- #
    #
    # ⚠️ 题目里**不许**提"他说了什么""语气如何" —— 本槽位是 ``visual``，
    #    只准看表情。schema §3.3 明确 ASR 情绪与它各自独立、可交叉校验，
    #    任何一侧都不覆盖另一侧。
    q["emo.visual"] = _choice(
        "只看面部表情与神态（眉毛、嘴角、眼睛），人物当前是什么情绪？"
        "不要参考他正在说什么、语气如何、穿什么。"
        "判据：皱眉/眉头紧锁/嘴角下压/目光锐利 ⇒ 不耐烦或愤怒；"
        "微笑、嘴角上扬 ⇒ 满意或高兴；脸颊放松、表情平和 ⇒ 平静。"
        "脸清晰可见时不要选 unknown。",
        EMOTION_LABELS)

    # ---- 4. BEHAVIOR -------------------------------------------------- #
    q["beh.posture"] = _choice("人物当前的姿态是？", POSTURE)
    q["beh.motion"] = _choice(
        "人物当前的运动状态是？参考多帧画面判断位移方向。", MOTION)
    q["beh.activity"] = _choice(
        "人物当前在进行什么持续活动？只在活动明确且持续时选择，短时手势不算。",
        ACTIVITY)
    q["beh.duration"] = _score(
        "这个姿态/运动/活动在本观测窗口内已经持续了多久？",
        BEHAVIOR_DURATION_LEVELS)

    # ---- 5. RELATION -------------------------------------------------- #
    q["rel.person_count"] = _choice(
        "画面里一共有几个人？", PERSON_COUNT)
    q["rel.interaction"] = _choice(
        "人物之间存在什么可观察的互动？", INTERACTION)
    q["rel.target"] = _choice(
        "主要人物当前在与什么交互？", REL_TARGET)

    # ---- 6. ENV ------------------------------------------------------- #
    if include_scene is None:
        include_scene = (stage == "full")
    if include_scene:
        q["env.scene"] = _choice("这是什么类型的场景？", SCENE)
    if include_lighting:
        q["env.lighting"] = _choice("画面整体光照情况如何？", LIGHTING)

    # ---- 7. POSITION -------------------------------------------------- #
    q["pos.region"] = _choice(
        "主要人物位于画面的哪个区域（把画面按 3×3 九宫格划分）？",
        POSITION_REGION)

    return q


#: 动作 id → 动作名（组装侧按 ``act.<name>`` 取）。
ACTION_QUESTION_IDS: Dict[str, str] = {
    f"act.{name}": name for name in ACTION_EVENTS
}


def question_ids(stage: str = "turn", tier2: bool = False) -> List[str]:
    """题集 id 列表（测试与日志用）。"""
    return list(build_questions(stage=stage, tier2=tier2).keys())
