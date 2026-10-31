from typing import Dict


def classify_ticket(query: str, risk_info: Dict) -> Dict:
    risk_types = risk_info.get("risk_types", [])

    if "battery_safety" in risk_types or "dangerous_repair" in risk_types:
        return {
            "category": "安全风险",
            "priority": "高",
            "department": "售后安全组"
        }

    if "refund_boundary" in risk_types:
        return {
            "category": "退换货/退款",
            "priority": "中",
            "department": "售后审核组"
        }

    if "complaint" in risk_types:
        return {
            "category": "投诉升级",
            "priority": "中",
            "department": "客户体验组"
        }

    return {
        "category": "普通咨询",
        "priority": "低",
        "department": "客服一线"
    }
