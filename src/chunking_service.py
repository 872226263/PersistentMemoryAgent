# src/chunking_service.py
import sys
import os
from typing import List
from loguru import logger
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 这确保了我们可以从src目录导入其他模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 必须先导入并初始化 model_manager 和配置
from .model_services import model_manager
from .config import settings

class ChunkingService:
    def __init__(self, embedding_model):
        if embedding_model is None:
            raise ValueError("ChunkingService需要一个有效的嵌入模型进行初始化。")
        logger.info("正在初始化 ChunkingService...")
        self.semantic_splitter = SemanticChunker(
            embeddings=embedding_model, 
            breakpoint_threshold_type="percentile" # 使用百分位数作为断点阈值，更具鲁棒性
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=int(settings.CHUNK_SIZE * 0.1), # 设置10%的重叠以保证上下文连续性
            length_function=len,
        )
        logger.success("ChunkingService 初始化成功，配置了 Semantic 和 Recursive 分割器。")

    def semantic_chunk_text(self, text: str) -> List[str]:
        """
        执行带容量保证的两阶段分块流程。
        1. 首先进行语义分块。
        2. 然后对超长的块进行递归字符分割，确保所有块都小于CHUNK_SIZE。
        """
        logger.info(f"开始对长度为 {len(text)} 的文本进行两阶段分块...")
        if not text.strip():
            logger.warning("输入文本为空，返回空列表。")
            return []
        
        # 阶段一：初步语义分块
        semantic_chunks = self.semantic_splitter.split_text(text)
        logger.info(f"第一阶段（语义）完成，产出 {len(semantic_chunks)} 个块。")

        # 阶段二：容量合规性检查与再分割
        final_chunks = []
        oversized_chunks_count = 0
        for chunk in semantic_chunks:
            if len(chunk) > settings.CHUNK_SIZE:
                oversized_chunks_count += 1
                sub_chunks = self.recursive_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        if oversized_chunks_count > 0:
            logger.info(f"第二阶段（容量保证）完成。{oversized_chunks_count} 个语义块被二次分割。")
        
        logger.success(f"两阶段分块完成，最终得到 {len(final_chunks)} 个合规的块。")
        return final_chunks

# --- 全局单例 ---
# 确保 model_manager 已经加载了模型
if not hasattr(model_manager, 'embedding_model') or model_manager.embedding_model is None:
    # 这是一个安全检查，正常情况下 model_manager 应该在应用启动时就绪
    logger.warning("ModelManager或其嵌入模型尚未初始化。ChunkingService可能无法正常工作。")
    chunking_service = None
else:
    # 创建 ChunkingService 的全局单例
    chunking_service = ChunkingService(model_manager.embedding_model) 