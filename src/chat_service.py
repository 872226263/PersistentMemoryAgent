# src/chat_service.py
import sys
import os
import asyncio
import logging
from typing import List, Dict, AsyncGenerator
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .memory_service import memory_service
from .model_services import model_manager
from .config import settings

class ChatService:
    def __init__(self):
        logger.info("ChatService initialized.")

    async def _archive_conversation_task(self, model_id: str, relevant_archive_summary_nodes: list, user_query: str, ai_response: str):
        """后台任务，根据上下文决定是追加记忆还是创建新档案。"""
        SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD
        
        if relevant_archive_summary_nodes and relevant_archive_summary_nodes[0].score >= SIMILARITY_THRESHOLD:
            archive_id_to_append = relevant_archive_summary_nodes[0].payload.get('archive_id')
            if archive_id_to_append is not None:
                logger.info(f"对话与档案 {archive_id_to_append} (通过摘要节点 {relevant_archive_summary_nodes[0].id}) 相关(得分: {relevant_archive_summary_nodes[0].score:.2f})，将追加记忆。")
                interaction = f"User: {user_query}\nAssistant: {ai_response}"
                await memory_service.add_interaction_to_archive(archive_id_to_append, interaction)
            else:
                logger.warning(f"相关摘要节点 {relevant_archive_summary_nodes[0].id} 的 payload 中缺少 archive_id，无法追加。将创建新档案。")
                interaction = f"User: {user_query}\nAssistant: {ai_response}"
                doc_name = user_query[:50]
                await memory_service.archive_memory(model_id, doc_name, interaction)
        else:
            score_info = f"最高得分: {relevant_archive_summary_nodes[0].score:.2f}" if relevant_archive_summary_nodes else "无相关摘要"
            logger.info(f"未找到相关对话 ({score_info})，将创建新的记忆档案。")
            interaction = f"User: {user_query}\nAssistant: {ai_response}"
            doc_name = user_query[:50]
            await memory_service.archive_memory(model_id, doc_name, interaction)

    async def generate_response(self, model_id: str, messages: List[Dict[str, str]], stream: bool = False):
        """
        生成AI回复的核心方法，编排完整的两阶段AAG循环和记忆归档。
        """
        user_query = messages[-1]['content']

        INSTRUCTION_PREFIX = "Based on the chat history"

        if user_query.strip().startswith(INSTRUCTION_PREFIX):
            logger.info("检测到内部指令，绕过记忆系统直接调用LLM...")
            return await model_manager.get_chat_response_async(messages, system_instruction=settings.SYSTEM_PROMPT, stream=stream)
        else:
            logger.info(f"接收到来自模型 '{model_id}' 的新查询: '{user_query}'")

            recalled_archive_summary_nodes = memory_service.search_top_level_summaries(
                model_id, user_query, top_k=settings.ARCHIVE_RECALL_TOP_K
            )

            rag_infused_messages = messages
            
            unique_archive_ids_from_summaries: Dict[int, float] = {}
            if recalled_archive_summary_nodes:
                for node_point in recalled_archive_summary_nodes:
                    aid = node_point.payload.get('archive_id')
                    if aid is not None:
                        if aid not in unique_archive_ids_from_summaries or node_point.score > unique_archive_ids_from_summaries[aid]:
                            unique_archive_ids_from_summaries[aid] = node_point.score
            
            if unique_archive_ids_from_summaries:
                logger.info(f"召回阶段：找到 {len(unique_archive_ids_from_summaries)} 个可能相关的档案 (基于顶层摘要)。")
                all_recalled_chunks = []
                for archive_id, score in sorted(unique_archive_ids_from_summaries.items(), key=lambda item: item[1], reverse=True):
                    logger.info(f"处理档案 {archive_id} (顶层摘要相关性: {score:.2f})")
                    chunks_in_archive = memory_service.search_chunks_in_archive(
                        archive_id, user_query, top_k=settings.CHUNK_RECALL_TOP_K
                    )
                    all_recalled_chunks.extend(chunks_in_archive)

                if all_recalled_chunks:
                    logger.info(f"召回阶段：共找到 {len(all_recalled_chunks)} 个候选文本块，进入重排阶段。")
                    
                    chunk_contents = [chunk.payload['content'] for chunk in all_recalled_chunks if 'content' in chunk.payload]
                    
                    if not chunk_contents:
                        logger.warning("召回的文本块中 payload 缺少 'content' 字段或内容为空。")
                    else:
                        reranked_indices_scores = model_manager.rerank(query=user_query, docs=chunk_contents)
                        
                        final_chunk_scored_points = []
                        for original_idx, rerank_score in reranked_indices_scores[:settings.RERANK_TOP_K]:
                            final_chunk_scored_points.append(all_recalled_chunks[original_idx])

                        mcp_context = "我回忆起了以下最相关的信息：\n"
                        for chunk_point in final_chunk_scored_points:
                            mcp_context += f"- {chunk_point.payload.get('content', 'Unavailable content')}\n"
                        logger.info(f"重排阶段：筛选出 {len(final_chunk_scored_points)} 个最相关的文本块构建内心独白。")
                        
                        rag_infused_messages = [
                            {"role": "assistant", "content": f"（{mcp_context}）"},
                        ] + messages
                else:
                    logger.warning(f"在已召回的档案中未找到任何相关的文本块。")
            else:
                logger.warning("在记忆中未找到相关的高层摘要，将作为常规聊天处理。")

            logger.info("正在生成最终回复...")
            llm_response = await model_manager.get_chat_response_async(rag_infused_messages, system_instruction=settings.SYSTEM_PROMPT, stream=stream)

            if stream:
                async def _archiving_stream_wrapper(response_stream):
                    full_response_content_parts = []
                    async for chunk_text_part in response_stream:
                        full_response_content_parts.append(str(chunk_text_part))
                        yield chunk_text_part
                    final_response = "".join(full_response_content_parts)
                    asyncio.create_task(self._archive_conversation_task(model_id, recalled_archive_summary_nodes, user_query, final_response))
                return _archiving_stream_wrapper(llm_response)
            else:
                await self._archive_conversation_task(model_id, recalled_archive_summary_nodes, user_query, llm_response)
                return llm_response

# 全局单例
chat_service = ChatService()