# src/server.py
import sys
import os
import uuid
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .memory_service import memory_service
from .chat_service import chat_service
from .config import settings
from loguru import logger
import uvicorn

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PersistentMemoryAgent - Dynamic Memory Forest API",
    version="3.0"
)

# --- Pydantic Models ---

# --- Archive Endpoint Models ---
class ArchiveRequest(BaseModel):
    model_id: str = Field(..., description="要归档到的AI模型的唯一标识符")
    document_name: str = Field(..., description="要归档的文档的名称")
    document_content: str = Field(..., description="要归档的文档的完整内容")

# --- Chat Endpoint Models (OpenAI-Compatible) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="正在交互的AI模型的唯一标识符")
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]

# --- MemoryNode Model (新增) ---
class MemoryNode(BaseModel):
    id: int
    summary: str
    content: str
    node_type: str # 你可以考虑使用 Literal['LEAF_CHUNK', 'SUMMARY_NODE'] 来更严格地定义
    relevance_score: float # TDD 中显示这是一个浮点数

# --- API Endpoints ---


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI兼容的聊天端点。
    所有复杂性（记忆检索、AAG循环）都被封装在ChatService中。
    """
    logger.info(f"接收到对模型 '{request.model}' 的聊天请求。")
    
    response_generator = await chat_service.generate_response(
        model_id=request.model,
        messages=[msg.model_dump() for msg in request.messages],
        stream=request.stream
    )

    if request.stream:
        async def sse_formatter(gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
            async for chunk_text in gen:
                stream_response = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(delta=DeltaMessage(content=chunk_text))]
                )
                yield f"data: {stream_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(sse_formatter(response_generator), media_type="text/event-stream")
    else:
        # 非流式响应：直接返回从ChatService获得的结果
        return {"model": request.model, "choices": [{"message": {"role": "assistant", "content": response_generator}}]}


@app.get("/tools/explore_memory_node", response_model=List[MemoryNode], tags=["MCP Tools"])
async def explore_memory_node(
    node_id: int = Query(..., description="要探索的父摘要节点的ID。"),
    query: str = Query(..., description="用户的原始问题，用于帮助排序找到最相关的细节。")
):
    # TODO: 实现此端点的具体逻辑
    # 例如:
    # children_nodes_data = await memory_service._arch ive_drill_down(node_id, query, threshold=0.5) # 假设阈值为0.5
    # return [MemoryNode(**node_data) for node_data in children_nodes_data] # 假设 _archive_drill_down 返回的数据可以直接转换
    logger.warning(f"Endpoint /tools/explore_memory_node called with node_id={node_id}, query='{query}' but is not yet implemented.")
    # 为了让服务器能启动，这里暂时返回一个空列表，它符合 List[MemoryNode] 类型
    return []





if __name__ == "__main__":
    logger.add("logs/server.log", rotation="10 MB", level="INFO")
    logger.info("启动 PersistentMemoryAgent API 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  