import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from config import MODEL_NAME, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from retriever import get_retriever

# 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1
)

# 问题改写提示（结合历史）
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据聊天历史，将用户问题改写为独立问题，使其在无需历史的情况下也可理解。"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

# 问答提示
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下上下文回答问题：\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建完整 RAG 链（检索器在每次调用时懒加载，避免启动时因向量库不存在而崩溃）
rag_chain = (
    {
        "context": (
            {"question": lambda x: x["question"], "history": lambda x: x.get("history", [])}
            | contextualize_q_prompt
            | llm
            | StrOutputParser()
            | (lambda x: get_retriever().invoke(x))
            | format_docs
        ),
        "question": lambda x: x["question"],
        "history": lambda x: x.get("history", [])
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 会话存储
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 具有消息历史的链
conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

def ask_question(question: str, session_id: str = "default"):
    return conversational_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )