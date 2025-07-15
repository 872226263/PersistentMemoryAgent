# tests/mocks.py
import asyncio
from typing import List, Dict, Tuple, AsyncGenerator, Any

# --- 模拟 ModelManager 的核心方法 ---

# 模拟 EmbeddingService (model_manager.get_embeddings)
def mock_get_embeddings(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    模拟生成嵌入向量。
    为每个文本返回一个固定长度和值的向量。
    """
    # print(f"[MockEmbeddingService] Generating embeddings for {len(texts)} texts. is_query={is_query}")
    # 使用一个简单的、可预测的嵌入向量，例如维度为3
    # 实际项目中，这里的维度应该与 initialize_db.py 中的 VECTOR_DIMENSION 一致，
    # 但对于模拟，只要形状正确即可。
    mock_embedding_dim = 1024 # 与 config.py 中的 VECTOR_DIMENSION 一致
    
    embeddings = []
    for i, text in enumerate(texts):
        # 创建一个基于文本长度和索引的简单确定性向量
        vec = [(float((len(text) + i + j) % 100) / 100.0) for j in range(mock_embedding_dim)]
        embeddings.append(vec)
    return embeddings

# 模拟 RerankerService (model_manager.rerank)
def mock_rerank(query: str, docs: list[str]) -> list[tuple[int, float]]:
    """
    模拟重排文档。
    简单地按输入顺序返回文档，并赋予递减的虚拟分数。
    """
    # print(f"[MockRerankerService] Reranking {len(docs)} docs for query: '{query}'")
    # 返回原始索引和模拟分数
    # 例如，可以简单地返回原始顺序，分数从1.0开始递减
    reranked_results = []
    for i, _ in enumerate(docs):
        reranked_results.append((i, 1.0 - (i * 0.1))) # (原始索引, 模拟分数)
    
    # 或者，为了更好地模拟，可以根据查询与文档的简单匹配度来打分
    # 这里我们保持简单，直接按顺序返回
    reranked_results.sort(key=lambda x: x[1], reverse=True) # 确保分数是降序的
    return reranked_results

# 模拟 LLMService (model_manager.summarize)
async def mock_summarize(text: str) -> str:
    """
    模拟生成摘要。
    返回一个包含原始文本前缀的固定格式摘要。
    """
    # print(f"[MockLLMService] Summarizing text: '{text[:50]}...'")
    await asyncio.sleep(0.01) # 模拟异步操作的微小延迟
    return f"Mocked Summary: {text[:min(len(text), 50)]}..."

# 模拟 LLMService (model_manager.get_chat_response_async)
async def mock_get_chat_response_async(
    messages: List[Dict[str, str]],
    system_instruction: str = None, # 模拟接收 system_instruction
    stream: bool = False
) -> str | AsyncGenerator[str, None]:
    """
    模拟获取聊天回复。
    - 非流式：返回一个基于最后一条用户消息的固定回复。
    - 流式：异步生成几个文本块。
    """
    user_query = "No user message found"
    if messages and messages[-1]['role'] == 'user':
        user_query = messages[-1]['content']
    
    # print(f"[MockLLMService] Generating chat response for query: '{user_query}', stream={stream}")
    # print(f"System instruction received: {system_instruction}")

    await asyncio.sleep(0.01) # 模拟异步操作的微小延迟

    if stream:
        async def stream_generator():
            response_parts = [
                f"Mock stream part 1 for: {user_query[:20]}...",
                f" Mock stream part 2 content.",
                f" End of mock stream."
            ]
            for part in response_parts:
                await asyncio.sleep(0.005) # 模拟流式块之间的延迟
                yield part
        return stream_generator()
    else:
        return f"Mocked non-stream response to: {user_query}"

# --- 用于 Patch 的 Mock ModelManager 实例 ---
# 在测试中，我们可以 patch 'src.model_services.model_manager' 的这些方法
# 例如: @patch('src.model_services.model_manager.get_embeddings', new=mock_get_embeddings)

class MockModelManager:
    """
    一个 ModelManager 的模拟类，可以用于更方便地 patch 整个对象。
    或者在测试设置中直接实例化和注入。
    """
    def __init__(self):
        self.embedding_model = "mock_embedding_model_instance"
        self.reranker_model = "mock_reranker_model_instance"
        self.chat_provider = "mock_chat_provider_instance"
        self.summary_provider = "mock_summary_provider_instance"
        self.device = "cpu"
        # 模拟 LLMProvider 中的 model_name
        if hasattr(self.chat_provider, 'model_name'):
            self.chat_provider.model_name = "mock-chat-model"
        if hasattr(self.summary_provider, 'model_name'):
            self.summary_provider.model_name = "mock-summary-model"


    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        return mock_get_embeddings(texts, is_query)

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        return mock_rerank(query, docs)

    async def summarize(self, text: str) -> str:
        return await mock_summarize(text)

    async def get_chat_response_async(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str = None,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        return await mock_get_chat_response_async(messages, system_instruction, stream=stream)

    # 如果 ModelManager 还有其他被 MemoryService 调用的方法，也应该在这里模拟
    # 例如，如果直接访问了 embedding_model 的属性
    # def get_vector_dimension(self) -> int: # 假设有这个方法
    #     return 1024 # 与 config.py 中的 VECTOR_DIMENSION 一致

# 可以创建一个全局的 mock_model_manager 实例，方便在测试中导入和使用
# from tests.mocks import mock_model_manager_instance
# mock_model_manager_instance = MockModelManager() 