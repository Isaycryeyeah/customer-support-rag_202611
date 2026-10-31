import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
ASK_URL = f"{API_BASE_URL}/ask"
FEEDBACK_URL = f"{API_BASE_URL}/feedback"

st.set_page_config(page_title="中文 3C 售后客服助手", layout="wide")

st.title("中文 3C 售后客服助手")
st.caption("售后资料问答 · 风险提示 · 人工处理建议")

if "result" not in st.session_state:
    st.session_state.result = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

question = st.text_area(
    "请输入你的售后问题",
    value="我的笔记本电池鼓包了，可以自己拆开吗？"
)

if st.button("提交问题"):
    with st.spinner("正在检索资料并生成回答..."):
        try:
            response = requests.post(ASK_URL, json={"question": question}, timeout=60)
            if response.status_code == 503:
                body = response.json()
                if body.get('answer_type') != 'service_unavailable':
                    response.raise_for_status()
                st.session_state.result = body
            else:
                response.raise_for_status()
                st.session_state.result = response.json()
            st.session_state.feedback_submitted = False
        except requests.RequestException as e:
            st.session_state.result = None
            st.error(f"请求失败：{e}")

data = st.session_state.result

if data:
    if data.get('answer_type') == 'service_unavailable':
        st.warning('回答服务暂时不可用，请稍后重试。下方为人工处理建议。')
    elif data.get('answer_type') == 'low_confidence':
        st.info('当前资料不足，建议联系人工客服核实。')
    elif data.get('answer_type') == 'safety_refusal':
        st.warning('该问题涉及安全风险，请查看安全提示并联系专业售后。')
    st.subheader("回答")
    st.write(data["answer"])

    st.subheader("风险识别")
    st.json(data["risk_info"])

    st.subheader("是否需要人工处理")
    st.write("是" if data["need_human"] else "否")

    if data.get("ticket"):
        st.subheader("工单建议")
        st.json(data["ticket"])

    st.subheader("这个回答对您有帮助吗？")

    if st.session_state.feedback_submitted:
        st.info("感谢您的反馈！")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 有帮助"):
                try:
                    requests.post(
                        FEEDBACK_URL,
                        json={"log_id": data["log_id"], "rating": 1},
                        timeout=10
                    ).raise_for_status()
                    st.session_state.feedback_submitted = True
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"反馈提交失败：{e}")

        with col2:
            if st.button("👎 没有帮助"):
                try:
                    requests.post(
                        FEEDBACK_URL,
                        json={"log_id": data["log_id"], "rating": -1},
                        timeout=10
                    ).raise_for_status()
                    st.session_state.feedback_submitted = True
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"反馈提交失败：{e}")

    st.subheader("检索资料")
    if data.get('answer_type') == 'service_unavailable':
        st.caption('以下资料来自检索，尚未生成基于资料的回答。')
    for idx, ctx in enumerate(data["contexts"], start=1):
        with st.expander(f"资料 {idx}: {ctx.get('title', '')}"):
            st.write(ctx["content"])
            st.json({
                "chunk_id": ctx.get("chunk_id"),
                "doc_id": ctx.get("doc_id"),
                "score": ctx.get("rerank_score")
            })
