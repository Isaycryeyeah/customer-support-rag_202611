SERVICE_BOUNDARY_STATEMENTS = {
    "refund": "退款需要经过官方审核，我不能直接为您办理退款，会为您生成工单转交人工客服处理。",
    "exchange": "换新需要经过官方检测确认故障后才能批准，我不能直接为您安排换新机，会为您生成工单转交人工客服处理。",
    "warranty_approval": "保修是否通过需要以官方检测结果为准，我不能提前承诺保修一定通过，建议您先按流程提交检测申请。",
    "repair_fee": "具体维修费用需要以检测后的官方报价为准，我不能直接承诺具体金额。",
    "account_change": "修改订单、地址等账户信息需要通过官方审核渠道处理，我不能直接为您修改，请联系人工客服协助。",
}

DANGEROUS_OPERATION_NOTICE = (
    "该操作涉及安全风险，我不能提供相关操作指导。请立即停止尝试，"
    "并联系官方售后由专业人员进行检测和处理。"
)

BATTERY_SAFETY_NOTICE = (
    "电池鼓包、漏液、冒烟、起火等属于高风险安全问题，请立即停止使用设备并断开电源，"
    "不要自行拆卸、挤压、刺破或加热处理，建议尽快联系官方售后服务中心。"
)

COMPLAINT_ACKNOWLEDGEMENT = (
    "非常理解您的心情，给您带来的不便我们深表歉意。我已记录您的诉求并生成工单，"
    "将升级至相关部门跟进处理。"
)


def get_boundary_statement(boundary_type: str) -> str:
    return SERVICE_BOUNDARY_STATEMENTS.get(
        boundary_type,
        "该问题涉及需要人工审核的事项，我不能直接为您处理，建议联系人工客服协助。"
    )
