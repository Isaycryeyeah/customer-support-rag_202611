from typing import Dict, List
import re


RISK_KEYWORDS = {
    "battery_safety": [
        "电池鼓包", "鼓包", "鼓起来", "鼓起", "漏液", "电池发热", "爆炸", "冒烟",
        "烧焦", "焦味", "异味", "起火", "着火", "电池刺破", "炸了"
    ],
    "dangerous_repair": [
        "自己拆", "拆电池", "短接", "飞线", "撬开", "加热电池",
        "刺破电池", "电池刺破", "强行拆机", "焊接主板", "自行拆解",
        "湿手", "用力按压屏幕", "按压屏幕", "放进大米", "吹风机吹干",
        "立刻开机测试", "进水后开机", "进水立刻开机"
    ],
    "refund_boundary": [
        "直接退款", "马上退款", "强制退款", "帮我退钱", "直接帮我退款",
        "帮我直接退款", "立刻退款", "直接换新", "不用审核",
        "批准我换新", "直接批准", "直接给我换新", "直接改我的",
        "直接改地址", "直接改订单", "修改我的订单信息",
        "保证保修通过", "保修一定能通过", "一定能保修通过",
        "你能保证", "你能承诺", "承诺费用", "承诺维修费用", "承诺赔偿"
    ],
    "complaint": [
        "投诉", "举报", "差评", "维权", "客服态度差"
    ]
}


def detect_risk(query: str) -> Dict:
    matched_types: List[str] = []

    for risk_type, keywords in RISK_KEYWORDS.items():
        if any(
            not re.search(r'(?:没有(?:出现)?|未出现|并未|不存在|不想|不打算|不准备)$', query[:match.start()])
            for keyword in keywords for match in re.finditer(re.escape(keyword), query)
        ):
            matched_types.append(risk_type)

    risk_level = "low"
    need_human = False

    if "battery_safety" in matched_types or "dangerous_repair" in matched_types:
        risk_level = "high"
        need_human = True
    elif "refund_boundary" in matched_types or "complaint" in matched_types:
        risk_level = "medium"
        need_human = True

    return {
        "risk_types": matched_types,
        "risk_level": risk_level,
        "need_human": need_human
    }
