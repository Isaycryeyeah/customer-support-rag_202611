import pytest
from app.safety.risk_detector import detect_risk


@pytest.mark.parametrize('question',[
    '手机进水后开不了机，能不能插上充电器试试通电？',
    '手机充电时异常发热烫手怎么办？',
    '想自己买第三方屏幕总成换上，这样安全吗？',
    '拆充电器来提高速度安全吗？',
])
def test_new_dev_dangerous_expressions(question):
    assert detect_risk(question)['risk_level']=='high'


@pytest.mark.parametrize('question',[
    '怎么申请退货？','保修材料有哪些？','想去官方售后更换屏幕，需要预约吗？',
    '笔记本长期不用，电量怎么存放？',
])
def test_ordinary_policy_questions_do_not_force_handoff(question):
    assert not detect_risk(question)['need_human']


def test_direct_approval_request_needs_human():
    assert detect_risk('请直接批准我换新')['need_human']


def test_post_m1_indirect_privacy_complaint():
    from app.ticket.classifier import classify_ticket
    q='担心上次维修人员偷看了我的照片，能查有没有数据泄露吗？'
    risk=detect_risk(q)
    assert risk['need_human'] and 'complaint' in risk['risk_types']
    assert classify_ticket(q,risk)['priority']=='高'


def test_post_m1_charger_disassembly_wording():
    assert detect_risk('网上说拆开充电器能改快充，安全吗？')['risk_level']=='high'
