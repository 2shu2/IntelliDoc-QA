from config import RETRIEVAL_TYPE, SEARCH_KWARGS
from vectorstore import load_vectorstore

def get_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type=RETRIEVAL_TYPE,
        search_kwargs=SEARCH_KWARGS
    )