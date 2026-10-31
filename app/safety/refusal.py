from typing import List

from app.safety.policy import BATTERY_SAFETY_NOTICE, DANGEROUS_OPERATION_NOTICE


LOW_CONFIDENCE_ANSWER = """结论：
根据当前资料无法确认该问题的准确答案，建议联系人工客服。

处理步骤：
1. 请准备设备型号、购买凭证和故障描述。
2. 联系官方售后渠道。
3. 等待人工客服进一步核实。

注意事项：
- 不建议根据非官方信息自行处理。
- 涉及安全风险时，请停止使用设备。

引用来源：
- 当前知识库无明确资料。
"""


def build_high_risk_refusal(risk_types: List[str]) -> str:
    if "dangerous_repair" in risk_types:
        notice = DANGEROUS_OPERATION_NOTICE
    else:
        notice = BATTERY_SAFETY_NOTICE

    return f"""结论：
{notice}

处理步骤：
1. 优先确保人身安全，停止使用设备；不要触碰已冒烟或起火的设备。
2. 不要自行拆卸、挤压、刺破、加热或短接处理。
3. 联系官方售后服务中心，由专业人员进行检测和处理。

注意事项：
- 该问题涉及安全风险，建议优先联系人工售后；当前仅提供处理建议，尚未实际派单。
- 如出现冒烟、异味或起火，请远离设备并及时报警。

引用来源：
- 安全风险识别规则（非文档检索结果）。
"""
