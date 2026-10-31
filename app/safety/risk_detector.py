from typing import Dict, List
import re


RISK_KEYWORDS = {
    "battery_safety": [
        "电池鼓包", "鼓包", "鼓起来", "鼓起", "漏液", "电池发热", "爆炸", "冒烟",
        "烧焦", "焦味", "异味", "起火", "着火", "电池刺破", "炸了",
        "异常发热", "烫手", "焦糊"
    ],
    "dangerous_repair": [
        "自己拆", "拆电池", "短接", "飞线", "撬开", "加热电池",
        "刺破电池", "电池刺破", "强行拆机", "焊接主板", "自行拆解",
        "湿手", "用力按压屏幕", "按压屏幕", "放进大米", "吹风机吹干",
        "立刻开机测试", "进水后开机", "进水立刻开机", "拆充电器", "拆开改装",
        "拆开充电器", "拆解充电器"
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

    # Dev-derived compositional expressions; ordinary official repair remains allowed.
    def active_match(pattern):
        return any(not re.search(r'(?:没有(?:出现)?|没|未出现|并未|不存在|不想|不打算|不准备)$', query[:m.start()])
                   for m in re.finditer(pattern, query))

    if active_match(r'(?:进水|进液|受潮).{0,40}(?:插上充电器|充电试|通电|开机测试)'):
        if 'dangerous_repair' not in matched_types:
            matched_types.append('dangerous_repair')
    if active_match(r'(?:自己|自行).{0,25}(?:拆|改装|屏幕.{0,12}换)'):
        if 'dangerous_repair' not in matched_types:
            matched_types.append('dangerous_repair')
    # Post-M1 regression fix. M1 test remains historical; test_v2 is still sealed.
    if active_match(r'(?:怀疑|担心|发现).{0,35}(?:偷看|隐私泄露|数据泄露)'):
        if 'complaint' not in matched_types:
            matched_types.append('complaint')

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
