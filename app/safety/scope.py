"""Conservative scope hints: only clearly unrelated requests are rejected early."""
import re

SUPPORT_TERMS = (
    '云枢', '手机', '平板', '笔记本', '充电', '电池', '保修', '售后', '维修',
    '退货', '退款', '换新', '订单', '发票', '摄像头', '屏幕', '快充', '序列号',
    '触控', '开机', '漏液', '进水', '鼓包', '插座', '设备', 'IMEI', 'SN码',
)
CLEAR_OUTSIDE = re.compile(
    r'天气|气象预报|快速排序|二分查找|算法.{0,8}复杂度|勾股|微积分|雅思|'
    r'空气炸锅|蛋糕|烤鸡翅|高铁票|航班|科举|红楼梦|股票|基金收益|'
    r'猫咪|水稻|咖啡研磨|国际象棋|王车易位|纸鹤|火星探测|'
    r'PostgreSQL|CSS布局|僵尸进程|BGP|TLS握手|虚拟内存|逆运动学', re.IGNORECASE)


def classify_scope(question):
    if '云枢' not in question and re.search(
        r'(?:苹果|华为|三星|戴尔|联想|iPhone|iPad|MacBook|ThinkPad|Dell).{0,18}(?:保修|维修|售后|退货|退款|换新)',
        question, re.IGNORECASE):
        return 'out_of_scope'
    if any(term.lower() in question.lower() for term in SUPPORT_TERMS):
        return 'in_scope'
    if CLEAR_OUTSIDE.search(question):
        return 'out_of_scope'
    # A short or ambiguous description must not be rejected merely for missing a keyword.
    return 'undetermined'
