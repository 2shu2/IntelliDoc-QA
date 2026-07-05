import os
# ⚠️ 必须在所有 HuggingFace 相关 import 之前设置，否则会连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import tempfile
import uuid
from loader import load_documents
from splitter import split_documents
from vectorstore import create_vectorstore
from chain import ask_question

st.set_page_config(page_title="IntelliDoc-QA", layout="wide")
st.title("📄 IntelliDoc-QA 多格式文档智能问答系统")

# 会话 ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 侧边栏：文档上传与处理
with st.sidebar:
    st.header("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF、Word、TXT、HTML",
        type=["pdf", "docx", "txt", "html"],
        accept_multiple_files=True
    )
    if st.button("处理文档"):
        if not uploaded_files:
            st.warning("请先上传文件")
        else:
            all_docs = []
            for file in uploaded_files:
                # 保存为临时文件
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                # 加载文档
                docs = load_documents(tmp_path)
                all_docs.extend(docs)
                os.unlink(tmp_path)  # 删除临时文件

            # 分割 & 向量化
            splits = split_documents(all_docs)
            if not splits:
                st.error("文档无有效文本内容。PDF 可能是扫描件/图片，请上传包含文字的可编辑文档。")
            else:
                create_vectorstore(splits)
                st.success(f"成功处理 {len(splits)} 个文档块，向量库已更新")

# 主界面：聊天
st.header("💬 对话问答")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用问答链
    try:
        answer = ask_question(prompt, st.session_state.session_id)
    except Exception as e:
        answer = f"❌ 错误: {str(e)}"

    # 添加助手消息
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)