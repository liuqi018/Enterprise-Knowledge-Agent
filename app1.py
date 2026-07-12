import time
import streamlit as st

from AIRAGAgent.agent.react_agent import ReactAgent
from AIRAGAgent.rag.vector_store import VectorStoreService

from db import (
    init_db,
    create_conversation,
    get_conversations,
    get_messages,
    save_message,
    delete_conversation,
    update_conversation_title,
    clear_messages
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Enterprise Knowledge Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Init DB
# =========================
init_db()

# =========================
# Style
# =========================
st.markdown("""
<style>
.main {padding-top: 1rem;}

.stApp {
    background-color: #f8fafc;
}

.header-container {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    margin-bottom: 20px;
}

.header-title {
    font-size: 34px;
    font-weight: bold;
}

.header-desc {
    opacity: 0.9;
    font-size: 15px;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Session Init
# =========================
if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()

if "current_chat" not in st.session_state:

    conversations = get_conversations()

    if conversations:
        st.session_state.current_chat = conversations[0]["id"]

    else:
        chat_id = create_conversation()
        st.session_state.current_chat = chat_id

current_chat_id = st.session_state.current_chat

messages = get_messages(current_chat_id)

# =========================
# Header
# =========================
st.markdown("""
<div class="header-container">
    <div class="header-title">
        🤖 Enterprise Knowledge Agent
    </div>
    <div class="header-desc">
        企业制度知识库 · RAG检索增强 · Agent工具调用 · 智能问答
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Stats
# =========================
vector_store = VectorStoreService()

try:
    doc_count = vector_store.count_documents()
except Exception:
    doc_count = 0

conversation_count = len(get_conversations())

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("📚 知识库文档", doc_count)

with c2:
    st.metric("💬 会话数量", conversation_count)

with c3:
    st.metric("🤖 Agent状态", "在线")

st.divider()

# =========================
# Sidebar
# =========================
with st.sidebar:

    st.markdown("## 💬 历史会话")

    if st.button(
        "➕ 新建会话",
        use_container_width=True
    ):

        new_chat_id = create_conversation()

        st.session_state.current_chat = new_chat_id

        st.rerun()

    st.divider()

    conversations = get_conversations()

    for chat in conversations:

        chat_id = chat["id"]
        title = chat["title"]

        col1, col2 = st.columns([5, 1])

        # 打开会话
        with col1:

            if st.button(
                f"📄 {title}",
                key=f"open_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat = chat_id

                st.rerun()

        # 删除会话
        with col2:

            if len(conversations) > 1:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}"
                ):

                    current_id = st.session_state.current_chat

                    delete_conversation(chat_id)

                    new_conversations = get_conversations()

                    if current_id == chat_id:

                        if new_conversations:

                            st.session_state.current_chat = (
                                new_conversations[0]["id"]
                            )

                        else:

                            new_chat_id = create_conversation()

                            st.session_state.current_chat = (
                                new_chat_id
                            )

                    st.rerun()

    st.divider()

    st.markdown("## ⚙️ 系统管理")

    st.info(
        f"""
📚 文档数量：{doc_count}

💬 会话数量：{conversation_count}

🤖 Agent状态：运行中
"""
    )

    if st.button(
        "🔄 加载/更新知识库",
        use_container_width=True
    ):

        with st.spinner("正在构建向量索引..."):

            before_count = vector_store.count_documents()

            vector_store.load_document(
                force=before_count == 0
            )

            after_count = (
                vector_store.count_documents()
            )

            st.session_state.agent = ReactAgent()

        st.success(
            f"知识库更新完成：{before_count} → {after_count}"
        )

    if st.button(
        "🗑️ 清空当前会话",
        use_container_width=True
    ):

        clear_messages(current_chat_id)

        st.rerun()

    st.divider()

    st.markdown("### 📌 示例问题")

    st.markdown("""
- 年假怎么申请？
- 病假需要哪些材料？
- 超过5000元采购怎么审批？
- 报销需要上传什么附件？
- 采购流程是什么？
""")

# =========================
# Chat History
# =========================
for msg in messages:

    avatar = (
        "👨"
        if msg["role"] == "user"
        else "🤖"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):

        st.markdown(
            msg["content"]
        )

# =========================
# User Input
# =========================
prompt = st.chat_input(
    "请输入您的问题..."
)

if prompt:

    # 第一条消息自动命名
    if len(messages) == 0:

        update_conversation_title(
            current_chat_id,
            prompt[:20]
        )

    # 显示用户消息
    with st.chat_message(
        "user",
        avatar="👨"
    ):

        st.markdown(prompt)

    # 保存用户消息
    save_message(
        current_chat_id,
        "user",
        prompt
    )

    # 重新获取最新历史
    latest_messages = get_messages(
        current_chat_id
    )

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        placeholder = st.empty()

        full_response = ""

        try:

            with st.spinner(
                "🔍 检索知识库并分析中..."
            ):

                res_stream = (
                    st.session_state.agent
                    .execute_stream(
                        prompt,
                        latest_messages
                    )
                )

                for chunk in res_stream:

                    full_response += chunk

                    placeholder.markdown(
                        full_response + "▌"
                    )

                    time.sleep(0.01)

                placeholder.markdown(
                    full_response
                )

        except Exception as e:

            full_response = (
                f"系统异常：{str(e)}"
            )

            placeholder.error(
                full_response
            )

    save_message(
        current_chat_id,
        "assistant",
        full_response
    )

    st.rerun()