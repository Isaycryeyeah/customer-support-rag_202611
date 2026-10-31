from typing import Dict


def build_ticket(
    query: str,
    answer: str,
    risk_info: Dict,
    ticket_info: Dict,
    *, need_human: bool, handoff_reason: str, answer_type: str
) -> Dict:
    return {
        "user_question": query,
        "category": ticket_info["category"],
        "priority": ticket_info["priority"],
        "suggested_department": ticket_info["department"],
        "risk_level": risk_info["risk_level"],
        "need_human": need_human,
        "handoff_reason": handoff_reason,
        "answer_type": answer_type,
        "status": "suggested",
        "summary": f"用户咨询：{query}。处理状态：{answer_type}。建议联系{ticket_info['department']}；此信息仅为工单建议，尚未实际派单。",
        "model_answer": answer
    }
