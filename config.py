import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek 模型
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# 文本分割
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# 向量存储 (使用 HuggingFace 免费嵌入，无需额外 API)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 也可改为 "text2vec-base-chinese" 支持中文
PERSIST_DIR = "./chroma_db"

# 检索设置
RETRIEVAL_TYPE = "mmr"   # "similarity" 或 "mmr"
SEARCH_KWARGS = {
    "k": 8,
    "fetch_k": 40,
    "lambda_mult": 0.5
}