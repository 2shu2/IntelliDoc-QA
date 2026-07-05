"""
IntelliDoc-QA FastAPI 后端
启动: python server.py
访问: http://localhost:8000
"""
import os
import uuid
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# HuggingFace 镜像（必须最先设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from loader import load_documents
from splitter import split_documents
from vectorstore import create_vectorstore
from chain import ask_question

# 会话存储
sessions: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭逻辑"""
    yield


app = FastAPI(title="IntelliDoc-QA API", lifespan=lifespan)


# ---------- API 模型 ----------
class QuestionRequest(BaseModel):
    question: str
    session_id: str


class QuestionResponse(BaseModel):
    answer: str
    session_id: str


class ProcessResponse(BaseModel):
    status: str
    chunks: int
    message: str


# ---------- API 路由 ----------
@app.post("/api/ask", response_model=QuestionResponse)
async def api_ask(req: QuestionRequest):
    """问答接口"""
    try:
        answer = ask_question(req.question, req.session_id)
        return QuestionResponse(answer=answer, session_id=req.session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=ProcessResponse)
async def api_upload(files: list[UploadFile] = File(...)):
    """文档上传与向量化"""
    if not files:
        raise HTTPException(status_code=400, detail="请选择文件")

    all_docs = []
    for file in files:
        suffix = os.path.splitext(file.filename or ".txt")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_documents(tmp_path)
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="无法从文档中提取文本")

    splits = split_documents(all_docs)
    if not splits:
        raise HTTPException(
            status_code=400,
            detail="文档无有效文本。PDF 可能是扫描件，请上传可编辑文档。"
        )

    create_vectorstore(splits)
    return ProcessResponse(
        status="success",
        chunks=len(splits),
        message=f"成功处理 {len(splits)} 个文档块"
    )


@app.get("/api/session")
async def api_session():
    """生成新会话"""
    sid = str(uuid.uuid4())
    sessions[sid] = sid
    return {"session_id": sid}


# ---------- 静态文件 ----------
# 挂载静态资源目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("\n  IntelliDoc-QA 服务启动中...")
    print("  访问: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
