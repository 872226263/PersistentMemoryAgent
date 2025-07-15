# src/memory_service.py
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import uuid
import asyncio
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from loguru import logger
from typing import List, Tuple, Dict, Any, Optional
import math

from .config import settings
from .model_services import model_manager
from .chunking_service import chunking_service

def _calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个嵌入向量之间的余弦相似度。"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def _is_valid_vector(vector: List[float], expected_dim: int) -> bool:
    """检查向量是否有效（列表，所有元素为有限浮点数，维度正确）"""
    if not isinstance(vector, list):
        logger.warning(f"向量不是列表类型: {type(vector)}")
        return False
    if len(vector) != expected_dim:
        logger.warning(f"向量维度错误: 得到 {len(vector)}, 期望 {expected_dim}")
        return False
    for i, x in enumerate(vector):
        if not (isinstance(x, float) and math.isfinite(x)):
            logger.warning(f"向量在索引 {i} 处包含无效值: {x} (类型: {type(x)})")
            return False
    return True

class MemoryService:
    def __init__(self):
        self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        logger.info("MemoryService initialized and configured for PostgreSQL and Qdrant (Unified Collection).")

    @contextmanager
    def _get_db_cursor(self, use_dict_cursor=True):
        """建立、管理并关闭一个PostgreSQL数据库连接和游标。"""
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                dbname=settings.POSTGRES_DB
            )
            if use_dict_cursor:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()

            logger.trace("PostgreSQL connection opened.")
            yield cursor
            conn.commit()
            logger.trace("PostgreSQL transaction committed.")
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                logger.trace("PostgreSQL connection closed.")

    async def _create_and_vectorize_node(
        self, 
        cursor,
        model_id: str,
        archive_id: int, 
        parent_id: Optional[int], 
        depth: int, 
        node_type: str,
        content: str, 
        summary: str
    ) -> Optional[int]:
        """
        在PostgreSQL中创建memory_node，为其内容或预设摘要生成向量（如果需要则先生成摘要），并存入Qdrant统一集合。
        返回创建的 node_id，如果失败则返回 None。
        """
        final_summary = summary

        if node_type == 'LEAF_CHUNK' and not final_summary:
            if not content:
                logger.warning(f"LEAF_CHUNK (archive {archive_id}) 内容为空，无法生成摘要。")
                final_summary = ""
            else:
                logger.info(f"为 LEAF_CHUNK (archive {archive_id}) 内容生成摘要...")
                final_summary = await model_manager.summarize_leaf_chunk(content)
        elif node_type == 'SUMMARY_NODE' and not final_summary:
            if not content:
                logger.warning(f"SUMMARY_NODE (archive {archive_id}) content为空，无法生成摘要。")
                final_summary = ""
            else:
                logger.warning(f"SUMMARY_NODE (archive {archive_id}) 预设摘要为空，尝试从其内容生成摘要。")
                final_summary = await model_manager.summarize_leaf_chunk(content)

        try:
            cursor.execute(
                """
                INSERT INTO memory_nodes (archive_id, parent_id, path, depth, node_type, content, summary)
                VALUES (%s, %s, NULL, %s, %s, %s, %s) RETURNING id
                """,
                (archive_id, parent_id, depth, node_type, content, final_summary)
            )
            
            node_result = cursor.fetchone()
            if not node_result or 'id' not in node_result:
                logger.error(f"插入节点到档案 {archive_id} 失败，类型: {node_type}, 内容: {content[:50]}...")
                return None
            node_id: int = node_result['id']
            
            path_value = f"{node_id}/"
            if parent_id is not None:
                cursor.execute("SELECT path FROM memory_nodes WHERE id = %s", (parent_id,))
                parent_path_result = cursor.fetchone()
                if parent_path_result and parent_path_result['path']:
                    path_value = parent_path_result['path'] + f"{node_id}/"
                else:
                    logger.warning(f"Node {node_id} 的父节点 {parent_id} 路径未找到，path 将设为 {node_id}/")

            cursor.execute(
                "UPDATE memory_nodes SET path = %s WHERE id = %s",
                (path_value, node_id)
            )
            logger.info(f"Node {node_id} (Type: {node_type}) created in PostgreSQL with path {path_value}.")

            text_to_embed = final_summary
            if not text_to_embed:
                logger.warning(f"Node {node_id} (Type: {node_type}) 的 final_summary 为空。尝试使用 content 进行嵌入。")
                text_to_embed = content

            if not text_to_embed:
                logger.error(f"Node {node_id} (Type: {node_type}) 无文本(summary或content)可嵌入。跳过Qdrant更新。")
                return node_id

            node_embedding_list = model_manager.get_embeddings([text_to_embed], is_query=False)
            if not node_embedding_list or not node_embedding_list[0]:
                logger.error(f"未能为节点 {node_id} (Type: {node_type}) 生成嵌入。跳过Qdrant更新。")
                return node_id
            
            node_embedding = node_embedding_list[0]

            if not _is_valid_vector(node_embedding, settings.VECTOR_DIMENSION):
                logger.error(f"节点 {node_id} (Type: {node_type}) 生成了无效向量。跳过Qdrant更新。")
                return node_id

            qdrant_point = PointStruct(
                id=node_id,
                vector=node_embedding,
                payload={
                    "archive_id": archive_id,
                    "model_id": model_id,
                    "node_type": node_type,
                    "depth": depth,
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=settings.NODES_VECTOR_COLLECTION_NAME,
                points=[qdrant_point]
            )
            logger.success(f"节点 {node_id} (Type: {node_type}) 向量已存入Qdrant集合 '{settings.NODES_VECTOR_COLLECTION_NAME}'.")
            return node_id

        except psycopg2.Error as db_err:
            logger.error(f"PostgreSQL error while creating/vectorizing node for archive {archive_id}: {db_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while creating/vectorizing node for archive {archive_id}: {e}")
            return None

    async def archive_memory(self, model_id: str, document_name: str, document_content: str):
        logger.info(f"开始为模型 '{model_id}' 执行层级摘要归档: '{document_name}' (Unified Collection)")

        if chunking_service is None:
            logger.error("ChunkingService 尚未初始化，无法进行文档分块。归档中止。")
            return {"error": "ChunkingService not available."}

        archive_id_val: Optional[int] = None
        try:
            with self._get_db_cursor(use_dict_cursor=True) as cursor:
                cursor.execute(
                    "INSERT INTO archives (model_id, name) VALUES (%s, %s) RETURNING id",
                    (model_id, document_name)
                )
                result = cursor.fetchone()
                if not result or 'id' not in result:
                    logger.error("Failed to create archive record in PostgreSQL.")
                    return None
                archive_id_val = result['id']
                logger.success(f"在PostgreSQL中创建了Archive记录, ID: {archive_id_val}")
        except Exception as e:
            logger.error(f"创建Archive记录时出错: {e}")
            return None
        
        if archive_id_val is None: return None

        initial_chunks_content = chunking_service.semantic_chunk_text(document_content)
        if not initial_chunks_content:
            logger.warning("文档分块后为空，归档中止。")
            return {"archive_id": archive_id_val, "message": "文档为空，只创建了Archive记录。"}

        l0_node_ids: List[int] = []
        try:
            with self._get_db_cursor(use_dict_cursor=True) as cursor:
                for chunk_text in initial_chunks_content:
                    created_node_id = await self._create_and_vectorize_node(
                        cursor=cursor,
                        model_id=model_id,
                        archive_id=archive_id_val,
                        parent_id=None,
                        depth=0,
                        node_type='LEAF_CHUNK',
                        content=chunk_text,
                        summary=""
                    )
                    if created_node_id:
                        l0_node_ids.append(created_node_id)
                    else:
                        logger.error(f"未能为块创建L0节点: {chunk_text[:50]}...")
                
                if not l0_node_ids:
                    logger.error(f"未能为档案 {archive_id_val} 创建任何 L0 节点。")
                    return {"archive_id": archive_id_val, "message": "未能创建任何有效的L0节点。"}
                
                logger.success(f"为档案 {archive_id_val} 成功创建并向量化 {len(l0_node_ids)} 个 L0 叶子节点。")

        except Exception as e:
            logger.error(f"处理L0叶子节点时发生错误 (Archive ID: {archive_id_val}): {e}")
            return None

        current_nodes_to_summarize_info: List[Dict[str, Any]] = []
        try:
            with self._get_db_cursor(use_dict_cursor=True) as cursor:
                for node_id in l0_node_ids:
                    cursor.execute("SELECT id, summary FROM memory_nodes WHERE id = %s", (node_id,))
                    node_data = cursor.fetchone()
                    if node_data and node_data['summary']:
                        current_nodes_to_summarize_info.append(node_data)
                    elif node_data:
                         logger.warning(f"L0 Node {node_id} 的摘要为空，可能无法参与后续层级摘要。")
        except Exception as e:
            logger.error(f"无法获取 L0 节点摘要进行层级归档: {e}")
            return None

        if not current_nodes_to_summarize_info:
            logger.warning(f"档案 {archive_id_val}: L0 节点无有效摘要，无法进行层级归档。")
            return {"archive_id": archive_id_val, "message": "L0节点已创建，但无摘要进行层级归档。"}

        parent_map: Dict[int, List[int]] = {} 
        current_level_parent_candidate_nodes_info = current_nodes_to_summarize_info
        current_depth = 0

        while len(current_level_parent_candidate_nodes_info) > 1:
            current_depth += 1
            logger.info(f"--- 进入第 {current_depth} 层摘要，当前候选父节点数量: {len(current_level_parent_candidate_nodes_info)} ---")
            
            next_level_pre_summary_inputs: List[Tuple[str, List[int], List[str]]] = []
            
            temp_chain_concatenated_summaries = ""
            temp_chain_child_ids: List[int] = []
            temp_chain_child_summary_texts: List[str] = []
            
            for i, node_info in enumerate(current_level_parent_candidate_nodes_info):
                current_summary_text = node_info['summary']
                current_node_id = node_info['id']

                if not temp_chain_concatenated_summaries:
                    temp_chain_concatenated_summaries = current_summary_text
                    temp_chain_child_ids = [current_node_id]
                    temp_chain_child_summary_texts = [current_summary_text]
                elif (len(temp_chain_concatenated_summaries) + len(current_summary_text)) <= settings.CHUNK_SIZE:
                    temp_chain_concatenated_summaries += f"\n---\n{current_summary_text}"
                    temp_chain_child_ids.append(current_node_id)
                    temp_chain_child_summary_texts.append(current_summary_text)
                else:
                    next_level_pre_summary_inputs.append((temp_chain_concatenated_summaries, temp_chain_child_ids, temp_chain_child_summary_texts))
                    temp_chain_concatenated_summaries = current_summary_text
                    temp_chain_child_ids = [current_node_id]
                    temp_chain_child_summary_texts = [current_summary_text]
            
            if temp_chain_concatenated_summaries:
                next_level_pre_summary_inputs.append((temp_chain_concatenated_summaries, temp_chain_child_ids, temp_chain_child_summary_texts))

            if len(next_level_pre_summary_inputs) == len(current_level_parent_candidate_nodes_info) and len(next_level_pre_summary_inputs) > 1 :
                logger.info(f"层级 {current_depth}: 摘要合并未减少节点数量 ({len(next_level_pre_summary_inputs)} 个输入 vs {len(current_level_parent_candidate_nodes_info)} 个候选)，层级摘要过程结束。")
                break
            
            if not next_level_pre_summary_inputs:
                logger.warning(f"层级 {current_depth}: 未能生成任何用于下一层摘要的输入。")
                break

            logger.info(f"层级 {current_depth}: 预摘要输入数量: {len(next_level_pre_summary_inputs)}。正在生成摘要...")

            newly_created_summary_nodes_info: List[Dict[str, Any]] = []
            
            summarize_tasks = [
                model_manager.summarize_parent_node(child_summaries_list=item[2]) 
                for item in next_level_pre_summary_inputs
            ]
            generated_new_parent_summaries = await asyncio.gather(*summarize_tasks)
            
            try:
                with self._get_db_cursor(use_dict_cursor=True) as cursor:
                    for i, new_parent_summary_text in enumerate(generated_new_parent_summaries):
                        original_concatenated_content_for_parent = next_level_pre_summary_inputs[i][0]
                        child_node_ids_for_this_parent = next_level_pre_summary_inputs[i][1]

                        new_summary_node_id = await self._create_and_vectorize_node(
                            cursor=cursor,
                            model_id=model_id,
                            archive_id=archive_id_val,
                            parent_id=None,
                            depth=current_depth,
                            node_type='SUMMARY_NODE',
                            content=original_concatenated_content_for_parent,
                            summary=new_parent_summary_text
                        )
                        if new_summary_node_id:
                            newly_created_summary_nodes_info.append({'id': new_summary_node_id, 'summary': new_parent_summary_text})
                            parent_map[new_summary_node_id] = child_node_ids_for_this_parent
                            
                            cursor.execute("SELECT path FROM memory_nodes WHERE id = %s", (new_summary_node_id,))
                            parent_path_result = cursor.fetchone()
                            new_parent_node_path = parent_path_result['path'] if parent_path_result and parent_path_result['path'] else f"{new_summary_node_id}/"

                            for child_node_id in child_node_ids_for_this_parent:
                                cursor.execute(
                                    "UPDATE memory_nodes SET parent_id = %s WHERE id = %s",
                                    (new_summary_node_id, child_node_id)
                                )
                                cursor.execute("SELECT path FROM memory_nodes WHERE id = %s", (child_node_id,))
                                child_original_path_segments = cursor.fetchone()['path'].strip('/').split('/')
                                new_child_path_suffix = child_original_path_segments[-1] + "/"

                                new_child_path = new_parent_node_path + new_child_path_suffix
                                cursor.execute("SELECT path FROM memory_nodes WHERE id = %s", (child_node_id,))
                                current_child_node_data = cursor.fetchone()
                                if current_child_node_data and current_child_node_data['path']:
                                    updated_child_path = f"{new_parent_node_path}{child_node_id}/"
                                    cursor.execute(
                                        "UPDATE memory_nodes SET path = %s WHERE id = %s",
                                        (updated_child_path, child_node_id)
                                    )
                                else:
                                     logger.warning(f"无法获取子节点 {child_node_id} 的原始路径用于更新。")

                            logger.info(f"新摘要节点 {new_summary_node_id} (深度 {current_depth}) 已创建。子节点 {child_node_ids_for_this_parent} 已更新。")
                        else:
                            logger.error(f"未能为内容创建摘要节点: {original_concatenated_content_for_parent[:50]}...")
            except Exception as e:
                logger.error(f"处理层级 {current_depth} 摘要节点时发生错误: {e}")
                current_level_parent_candidate_nodes_info = []
                break

            if not newly_created_summary_nodes_info:
                logger.warning(f"层级 {current_depth}: 未能成功创建任何新的摘要节点。")
                break
            
            current_level_parent_candidate_nodes_info = newly_created_summary_nodes_info
            logger.success(f"层级 {current_depth}: 成功创建并向量化 {len(current_level_parent_candidate_nodes_info)} 个摘要节点。")

            if len(current_level_parent_candidate_nodes_info) <= 1:
                logger.info(f"层级摘要完成，最终剩余 {len(current_level_parent_candidate_nodes_info)} 个顶层摘要节点。")
                break
        
        logger.info(f"文档 '{document_name}' (Archive ID: {archive_id_val}) 层级归档流程完成。")
        return {"archive_id": archive_id_val, "message": "文档层级归档成功 (Unified Collection)."}

    def search_top_level_summaries(self, model_id: str, query: str, top_k: int = 5):
        """
        在指定模型的记忆中搜索顶层/高层摘要节点。
        """
        logger.info(f"在模型 {model_id} 的顶层摘要中搜索查询: '{query}'")
        query_embedding_list = model_manager.get_embeddings([query], is_query=True)
        if not query_embedding_list or not _is_valid_vector(query_embedding_list[0], settings.VECTOR_DIMENSION):
            logger.error("查询嵌入生成失败或无效。")
            return []
        query_embedding = query_embedding_list[0]
        
        search_results = self.qdrant_client.search(
            collection_name=settings.NODES_VECTOR_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="model_id", match=models.MatchValue(value=model_id)),
                    models.FieldCondition(key="node_type", match=models.MatchValue(value='SUMMARY_NODE')),
                ]
            ),
            limit=top_k,
        )
        
        logger.info(f"顶层摘要搜索完成，找到 {len(search_results)} 个相关摘要节点。")
        return search_results

    def search_chunks_in_archive(self, archive_id: int, query: str, top_k: int = 10):
        """在指定的档案ID内进行文本块(LEAF_CHUNK)的向量搜索"""
        logger.info(f"在档案 {archive_id} 的文本块中搜索查询: '{query}' (Unified Collection)")
        query_embedding_list = model_manager.get_embeddings([query], is_query=True)
        if not query_embedding_list or not _is_valid_vector(query_embedding_list[0], settings.VECTOR_DIMENSION):
            logger.error("查询嵌入生成失败或无效。")
            return []
        query_embedding = query_embedding_list[0]

        search_results = self.qdrant_client.search(
            collection_name=settings.NODES_VECTOR_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="archive_id", match=models.MatchValue(value=archive_id)),
                    models.FieldCondition(key="node_type", match=models.MatchValue(value='LEAF_CHUNK'))
                ]
            ),
            limit=top_k
        )
        
        logger.info(f"档案内文本块搜索完成，找到 {len(search_results)} 个相关文本块。")
        return search_results

    async def add_interaction_to_archive(self, archive_id: int, interaction_content: str):
        """
        (异步) 将新交互追加到存档。将作为新的L0叶子节点，并为其内容生成摘要。
        """
        logger.info(f"正在向档案ID {archive_id} 追加新交互 (Unified Collection)...")
        
        model_id = None
        try:
            with self._get_db_cursor(use_dict_cursor=True) as cursor:
                cursor.execute("SELECT model_id FROM archives WHERE id = %s", (archive_id,))
                archive_meta = cursor.fetchone()
                if not archive_meta:
                    logger.error(f"找不到 Archive ID {archive_id}，无法追加交互。")
                    return
                model_id = archive_meta['model_id']
        except Exception as e:
            logger.error(f"获取 Archive {archive_id} 元数据时出错: {e}")
            return

        if not model_id: return

        try:
            with self._get_db_cursor(use_dict_cursor=True) as cursor:
                new_node_id = await self._create_and_vectorize_node(
                    cursor=cursor,
                    model_id=model_id,
                    archive_id=archive_id,
                    parent_id=None,
                    depth=0, 
                    node_type='LEAF_CHUNK',
                    content=interaction_content,
                    summary=""
                )
                if new_node_id:
                    logger.success(f"新交互已作为节点 {new_node_id} (Archive {archive_id}) 存入并向量化。")
                else:
                    logger.error(f"未能为 Archive {archive_id} 创建新的交互节点。")
        except Exception as e:
            logger.error(f"向档案 {archive_id} 添加交互时出错: {e}")

    def get_node_details_by_ids(self, node_ids: List[int]) -> List[Dict[str, Any]]:
        """根据节点ID列表从PostgreSQL获取节点详细信息。"""
        if not node_ids:
            return []
        try:
            with self._get_db_cursor() as cursor:
                query = "SELECT id, archive_id, parent_id, path, depth, node_type, content, summary, created_at FROM memory_nodes WHERE id IN %s"
                cursor.execute(query, (tuple(node_ids),))
                nodes_data = cursor.fetchall()
                return nodes_data if nodes_data else []
        except Exception as e:
            logger.error(f"获取节点详情失败 (IDs: {node_ids}): {e}")
            return []

# 全局单例
memory_service = MemoryService()