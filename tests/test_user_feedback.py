from unittest.mock import Mock
import pytest

from app.generation.citations import format_citations,group_sources
from app.main_pipeline import CustomerSupportRAG
from app.safety.refusal import build_high_risk_refusal


@pytest.mark.parametrize('question',[
    '快速排序平均复杂度如何推导？','今天天气怎么样？','苹果MacBook电池的官方保修期是多少？',
])
def test_clear_outside_does_not_create_after_sales_ticket(question):
    retrieval,reranker,llm=Mock(),Mock(),Mock()
    r=CustomerSupportRAG(retrieval,reranker,llm).answer(question)
    assert r['domain_scope']=='out_of_scope' and r['answer_type']=='low_confidence'
    assert r['need_human'] is False and r['ticket'] is None
    retrieval.search.assert_not_called();llm.generate_safe.assert_not_called()


def test_ambiguous_or_in_domain_missing_evidence_still_has_help():
    retrieval,reranker,llm=Mock(),Mock(),Mock()
    retrieval.search.return_value=[]
    for q in ('它突然用不了了怎么办？','云枢手机出现资料没记录的罕见故障'):
        r=CustomerSupportRAG(retrieval,reranker,llm).answer(q)
        assert r['domain_scope']!='out_of_scope' and r['need_human']
        assert r['ticket']['need_human']


def test_charger_template_contains_reviewed_safety_details():
    answer=build_high_risk_refusal(['battery_safety'],'充电器插头发烫，插座冒烟，非官方快充有风险吗？')
    for point in ('安全操作','切断电源','不要反复插拔','兼容性风险','不要触碰'):
        assert point in answer


def contexts():
    return [
        {'title':'保修政策','source_path':'warranty.md','section':'期限','chunk_id':'w1'},
        {'title':'保修政策','source_path':'warranty.md','section':'范围','chunk_id':'w2'},
        {'title':'凭证说明','source_path':'invoice.md','section':'材料','chunk_id':'i1'},
    ]


@pytest.mark.parametrize('heading',['引用来源：','**引用来源**：','## 引用来源'])
def test_one_document_entry_keeps_distinct_passage_anchors(heading):
    raw='结论：期限见[资料1]，范围见[资料2]，凭证见[资料3]。\n\n'+heading+'\n- 保修政策\n- 保修政策\n- 凭证说明'
    answer,sources,status=format_citations(raw,contexts())
    assert len(sources)==2 and len(sources[0]['passages'])==2
    assert answer.count('保修政策')==1
    assert '[资料1]' in answer and '[资料2]' in answer
    assert '期限；范围' in answer and status['status']=='valid'


def test_source_list_is_not_mistaken_for_body_citations():
    answer,sources,status=format_citations('结论：[资料1]\n引用来源：\n- 资料2\n- 资料3',contexts())
    assert len(sources)==1 and sources[0]['passages'][0]['passage_id']==1
    assert len(sources[0]['passages'])==1


def test_invalid_or_missing_citations_are_not_invented():
    _,sources,status=format_citations('结论：[资料99]',contexts())
    assert status['status']=='invalid' and status['invalid_passage_ids']==[99]
    assert not any(s['cited'] for s in sources)
    answer,_,status=format_citations('结论：无引用文本',contexts())
    assert status['status']=='missing' and '正文未标明具体引用' in answer
