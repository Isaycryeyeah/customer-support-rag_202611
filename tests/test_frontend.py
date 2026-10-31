from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('status,answer_type', [(503, 'service_unavailable'), (200, 'low_confidence')])
def test_structured_failure_or_refusal_remains_visible(monkeypatch, status, answer_type):
    body = {'log_id': 1, 'answer_type': answer_type, 'answer': '请联系人工售后',
            'risk_info': {'risk_level': 'low'}, 'need_human': True,
            'ticket': {'need_human': True, 'status': 'suggested'}, 'contexts': []}
    response = Mock(status_code=status)
    response.json.return_value = body
    monkeypatch.setattr(requests, 'post', Mock(return_value=response))
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).run()
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state['result']['answer_type'] == answer_type
    assert not page.error
    if status == 503:
        assert any('暂时不可用' in warning.value for warning in page.warning)

