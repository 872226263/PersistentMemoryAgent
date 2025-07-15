# src/initialize_db.py
import psycopg2 # 为 PostgreSQL 导入
from psycopg2 import OperationalError
from qdrant_client import QdrantClient, models
from loguru import logger
import os # 仍然用于日志路径

# 从 src 目录导入 config
from .config import settings

def initialize_database():
    """
    初始化或重新初始化 PostgreSQL 数据库和 Qdrant 向量存储。
    - 如果 PostgreSQL 表 (`archives`, `memory_nodes`) 不存在，则创建它们。
    - 重新创建 Qdrant 向量集合 (`archives_collection`, `chunks_collection`)。
    """
    logger.info("开始数据库和向量存储初始化...")

    # --- 1. PostgreSQL 初始化 ---
    conn = None  # 在 try 块外部初始化 conn
    cursor = None # 在 try 块外部初始化 cursor
    try:
        logger.info(f"正在连接到 PostgreSQL 数据库位于 {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}...")
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB
        )
        cursor = conn.cursor()
        logger.success("成功连接到 PostgreSQL。")

        # 创建 `archives` 表 (根据 TDD)
        # `archives` 表：代表一个"档案"，即一个完整的"记忆树"
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS archives (
            id SERIAL PRIMARY KEY,
            model_id TEXT NOT NULL,          -- 关联的AI模型的唯一标识符
            name TEXT NOT NULL,              -- 档案的名称 (例如 "Project Phoenix Q3 Report")
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """)
        logger.success("表 'archives' 创建成功或已存在。")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archives_model ON archives (model_id);")
        logger.success("表 'archives' 的索引 'idx_archives_model' 创建成功或已存在。")


        # 创建 `memory_nodes` 表 (根据 TDD)
        # `memory_nodes` 表：存储一个记忆档案（树）中的所有节点
        # 注意：旧的 TDD 对 memory_nodes 使用 `summary TEXT NOT NULL`，
        # 但是 memory_service.py 在存储 L0 chunks 时，summary 是空字符串。
        # 为了保持一致性，我们将允许 summary 为 NULL 或给它一个默认值。
        # 我目前将保持 NOT NULL，这可能需要在 memory_service 中进行调整。
        # TDD 也提到了 `memory_nodes` 表的 `summary TEXT NOT NULL`。让我们遵循这一点。
        # SQLite 的 `initialize_db.py` 对 chunks 有 `summary TEXT NOT NULL`。
        # 让我们与 TDD 的 `memory_nodes` 保持一致。
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id SERIAL PRIMARY KEY,
            archive_id INTEGER NOT NULL,
            parent_id INTEGER,
            path TEXT,                         -- 物化路径 (例如 "1/5/12/")
            depth INTEGER,
            node_type TEXT NOT NULL CHECK(node_type IN ('LEAF_CHUNK', 'SUMMARY_NODE')),
            content TEXT NOT NULL,
            summary TEXT NOT NULL,             -- 内容的摘要
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (archive_id) REFERENCES archives (id) ON DELETE CASCADE,
            FOREIGN KEY (parent_id) REFERENCES memory_nodes(id) ON DELETE SET NULL
        );
        """)
        logger.success("表 'memory_nodes' 创建成功或已存在。")
        
        # 为 `memory_nodes` 创建索引 (根据 TDD)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_archive_parent ON memory_nodes (archive_id, parent_id);")
        logger.success("表 'memory_nodes' 的索引 'idx_nodes_archive_parent' 创建成功或已存在。")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_archive_path ON memory_nodes (archive_id, path);") # 如果查询特定档案，path 隐式包含 archive_id
        logger.success("表 'memory_nodes' 的索引 'idx_nodes_archive_path' 创建成功或已存在。")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_archive_depth ON memory_nodes (archive_id, depth);")
        logger.success("表 'memory_nodes' 的索引 'idx_nodes_archive_depth' 创建成功或已存在。")
        # 为 node_type 创建索引，因为我们会基于它进行过滤
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_node_type ON memory_nodes (node_type);")
        logger.success("表 'memory_nodes' 的索引 'idx_nodes_node_type' 创建成功或已存在。")

        conn.commit()
    except OperationalError as e:
        logger.error(f"PostgreSQL 连接失败: {e}")
        logger.error("请确保 PostgreSQL 服务正在运行，并且 .env 文件中的连接配置正确。")
        return
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL 初始化失败: {e}")
        if conn:
            conn.rollback() # 如果在执行查询时发生错误，则回滚
        return
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logger.info("PostgreSQL 连接已关闭。")

    # --- 2. Qdrant 初始化 ---
    # 这部分基本保持不变，但我们将使用 settings.VECTOR_DIMENSION
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        logger.info(f"成功连接到 Qdrant 位于 {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

        vector_size = settings.VECTOR_DIMENSION # 从配置中获取
        logger.info(f"为 Qdrant 集合使用向量维度: {vector_size}")

        # 创建统一的向量集合
        client.recreate_collection(
            collection_name=settings.NODES_VECTOR_COLLECTION_NAME, # 使用新的统一集合名称
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            # 可以考虑在这里为 payload 字段创建索引以优化过滤性能
            # payload_schema={
            #     "model_id": models.PayloadSchemaType.KEYWORD,
            #     "archive_id": models.PayloadSchemaType.INTEGER,
            #     "node_type": models.PayloadSchemaType.KEYWORD,
            #     "depth": models.PayloadSchemaType.INTEGER 
            # }
        )
        logger.success(f"成功重新创建 Qdrant 集合: '{settings.NODES_VECTOR_COLLECTION_NAME}'")

    except Exception as e:
        logger.error(f"Qdrant 初始化失败: {e}")
        logger.error("请确保 Qdrant 服务正在运行并且网络连接正常。")
        return

    logger.info("数据库和向量存储初始化完成！")

if __name__ == "__main__":
    # 如果 logs 目录不存在，则创建它
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(os.path.join(log_dir, "initialize_db.log"), rotation="10 MB", level="INFO")
    
    # 加载 .env 变量 (如果 initialize_db.py 作为主脚本运行)
    # 这通常在导入 `settings` 时由 pydantic-settings 处理，
    # 但确保一下没有坏处。
    from dotenv import load_dotenv
    load_dotenv()
    
    initialize_database()