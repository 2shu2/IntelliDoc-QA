# IntelliDoc-QA

基于 **LangChain + DeepSeek + Chroma** 的企业级多格式文档智能问答系统（RAG）。

上传 PDF / Word / TXT / HTML 文档，自动向量化入库，自然语言提问即可从文档中检索内容并生成回答，支持多轮对话。前端采用 **React + Tailwind CSS** 实现 cinematic glass-morphism 风格界面，后端基于 **FastAPI** 提供 RESTful API。

## 功能特性

- **多格式文档加载**：PDF / Word / TXT / HTML，按后缀自动选择加载器
- **递归文本分割**：chunk_size=1500, overlap=200，中文分隔符优化
- **本地向量存储**：HuggingFace Embeddings（all-MiniLM-L6-v2，免费）+ Chroma 持久化
- **MMR 检索**：最大边际相关性算法，k=8，平衡相关性与多样性
- **多轮对话**：LCEL 链 + 历史感知（问题改写 → 检索 → 上下文增强生成）
- **Cinematic UI**：React + Tailwind CSS 全屏视频背景 + 玻璃态（glass-morphism）设计

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│  Frontend: React + Tailwind (static/index.html)         │
│  Glass-morphism UI / Video Backgrounds / Responsive     │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────┐
│  Backend: FastAPI (server.py)                           │
│                                                         │
│  POST /api/upload  → loader.py → splitter.py            │
│                    → vectorstore.py (HuggingFace → Chroma)│
│                                                         │
│  POST /api/ask     → chain.py                           │
│    Step1: 问题改写（LLM + 对话历史）                       │
│    Step2: MMR 向量检索（retriever.py, Top-4）            │
│    Step3: 上下文增强生成（DeepSeek LLM）                  │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
├── server.py           # FastAPI 后端入口（启动这个）
├── static/
│   └── index.html      # React + Tailwind 前端（单文件）
├── config.py           # 全局配置（LLM/Embedding/分块/检索参数）
├── loader.py           # 多格式文档加载器
├── splitter.py         # 文本分割
├── vectorstore.py      # Chroma 向量库创建与加载
├── retriever.py        # MMR 检索器
├── chain.py            # RAG 链（问题改写 + 检索 + 生成 + 多轮记忆）
├── requirements.txt    # 依赖清单
├── .env.example        # 环境变量模板（安全）
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/2shu2/IntelliDoc-QA.git
cd IntelliDoc-QA
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key（[获取地址](https://platform.deepseek.com)）：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

### 4. 启动

```bash
python server.py
```

浏览器访问 `http://localhost:8000`，上传文档后即可对话问答。

## API 接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/upload` | POST | 上传文档（PDF/Word/TXT/HTML），自动向量化 |
| `/api/ask` | POST | 提问并获取回答 `{"question":"...", "session_id":"..."}` |
| `/api/session` | GET | 获取新会话 ID |

## 配置说明

所有参数在 `config.py` 中可修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `deepseek-v4-pro` | DeepSeek 模型 |
| `CHUNK_SIZE` | 1500 | 文本块大小（字符） |
| `CHUNK_OVERLAP` | 200 | 块之间重叠字符数 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 嵌入模型（可换 `text2vec-base-chinese`） |
| `RETRIEVAL_TYPE` | `mmr` | 检索方式（mmr / similarity） |
| `k` | 8 | 检索返回结果数 |

## 注意事项

- **首次运行**会从 HuggingFace 镜像下载嵌入模型（~90MB），需保持网络畅通
- 国内用户已默认配置 `hf-mirror.com` 镜像（在 `server.py` 第一行）
- 向量库保存在 `chroma_db/`，删除该目录可清空知识库
- 对话历史**仅在当前进程有效**，重启后清空
- 每次上传文档会**完全重建**向量库（非增量）

## License

MIT
