import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, PERSIST_DIR

def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

def load_vectorstore():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"向量库不存在: {PERSIST_DIR}\n"
            '请先在侧边栏上传文档并点击"处理文档"。'
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)