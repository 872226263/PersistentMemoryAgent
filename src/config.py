# src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- 模型路径 ---
    EMBEDDING_MODEL_PATH: str = "Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL_PATH: str = "Qwen/Qwen3-Reranker-0.6B"

    # --- LLM 提供商配置 ---
    CHAT_MODEL_PROVIDER: str = "gemini"
    CHAT_MODEL_NAME: str = "gemini-2.5-flash-preview-05-20"
    SYSTEM_PROMPT: str = "You are H.E.R., your primary objective is to meticulously understand and strictly adhere to every instruction I provide. Pay close attention to all details I give, ensuring your response perfectly aligns with my requirements. Prioritize my specific directives, even if it means overriding your typical response patterns or general knowledge."
    GOOGLE_API_KEY: str = "YOUR_GOOGLE_API_KEY_HERE"

    SUMMARY_MODEL_PROVIDER: str = "volcangine"
    SUMMARY_MODEL_NAME: str = "deepseek-v3-250324"
    VOLCANGINE_ARK_API_KEY: str = "YOUR_VOLCANGINE_ARK_API_KEY_HERE"
    
    LLM_RETRY_ATTEMPTS: int = 10
    LLM_INITIAL_RETRY_DELAY: float = 1.0

    HTTP_PROXY: str = None

    # 调整为针对统一集合的检索参数
    ARCHIVE_RECALL_TOP_K: int = 5 # 用于在摘要节点中检索档案
    CHUNK_RECALL_TOP_K: int = 20 # 用于在叶子节点中检索块
    RERANK_TOP_K: int = 5

    CHUNK_SIZE: int = 512
    SIMILARITY_THRESHOLD: float = 0.75 # ChatService 中的归档决策阈值

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # 统一的 Qdrant 集合名称，用于存储所有 memory_nodes 的向量
    NODES_VECTOR_COLLECTION_NAME: str = "her_memory_nodes" # 新名称，替换旧的两个

    # --- PostgreSQL 数据库配置 ---
    # DB_PATH: str = "memory.db" # 如果使用 PostgreSQL 存储主要元数据，则不再相关
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "her_user"
    POSTGRES_PASSWORD: str = "her_password"
    POSTGRES_DB: str = "her_memory_db"
    
    # --- 向量维度 (TODO: 理想情况下从模型获取，但对于 init_db 可以在此处设置) ---
    VECTOR_DIMENSION: int = 1024 # 如果你的嵌入模型有不同的维度，请调整

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'allow'

settings = Settings()