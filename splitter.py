from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(docs):
    # 过滤掉空文档（扫描件、图片PDF等无文字内容）
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    splits = splitter.split_documents(docs)
    # 过滤空块
    return [s for s in splits if s.page_content.strip()]