# src/model_services.py
import sys
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
import json
from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .config import settings
# 导入新的提供商类
from .llm_providers import GoogleGeminiProvider, VolcangineDeepSeekProvider, LLMProvider

# --- Pydantic模型定义 (用于解析LLM的JSON输出) ---
class LeafSummaryOutput(BaseModel):
    summary: str = Field(description="生成的叶子节点摘要文本")

class ParentSummaryOutput(BaseModel):
    summary: str = Field(description="生成的父级摘要文本")

# --- TDD Prompt 文本定义 (为清晰起见，直接从TDD文档复制核心部分) ---

# PROMPT_CREATE_LEAF_SUMMARY (用于叶子节点)
PROMPT_LEAF_SUMMARY_ROLE = "你是一个严谨的信息归档员。"
PROMPT_LEAF_SUMMARY_TASK = "为下面提供的原始文本块生成一个简洁、客观、信息密集的摘要。这个摘要将作为未来检索的唯一依据，因此必须准确反映文本的核心内容。"
PROMPT_LEAF_SUMMARY_CONSTRAINTS = {
    "max_length_chars": 200,
    "style": "中立、事实性",
    "language": "与输入文本语言保持一致"
}

# PROMPT_CREATE_PARENT_SUMMARY (用于父节点)
PROMPT_PARENT_SUMMARY_ROLE = "你是一个高级知识整合专家，擅长从多个相关的信息片段中提炼出更高层次的共同主题。"
PROMPT_PARENT_SUMMARY_TASK = "下面提供了多个子节点的摘要。请将它们融会贯通，生成一个能够概括所有子节点核心内容的、更具抽象性的父级摘要。不要简单地拼接，而是要进行真正的综合与提炼。"
PROMPT_PARENT_SUMMARY_CONSTRAINTS = {
    "max_length_chars": 200,
    "focus": "寻找共性、上层概念或流程关系",
    "language": "与输入文本语言保持一致"
}

# --- Parsers ---
LEAF_SUMMARY_PARSER = PydanticOutputParser(pydantic_object=LeafSummaryOutput)
PARENT_SUMMARY_PARSER = PydanticOutputParser(pydantic_object=ParentSummaryOutput)

# --- Langchain Prompt Templates ---

# 叶子节点摘要模板
leaf_system_content = (
    f"{PROMPT_LEAF_SUMMARY_ROLE}\n"
    f"{PROMPT_LEAF_SUMMARY_TASK}\n"
    f"约束: {json.dumps(PROMPT_LEAF_SUMMARY_CONSTRAINTS, ensure_ascii=False)}"
)
# 注意：input_variables 中的 "text_chunk" 对应 HumanMessagePromptTemplate.from_template 中的 {text_chunk}
# format_instructions 会被部分变量填充
leaf_human_template_str = (
    "原始文本:\n```\n{text_chunk}\n```\n\n"
    "请严格按照以下格式指令进行输出:\n{format_instructions}"
)
LEAF_SUMMARY_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(leaf_system_content),
    HumanMessagePromptTemplate.from_template(leaf_human_template_str)
]).partial(format_instructions=LEAF_SUMMARY_PARSER.get_format_instructions())


# 父节点摘要模板
parent_system_content = (
    f"{PROMPT_PARENT_SUMMARY_ROLE}\n"
    f"{PROMPT_PARENT_SUMMARY_TASK}\n"
    f"约束: {json.dumps(PROMPT_PARENT_SUMMARY_CONSTRAINTS, ensure_ascii=False)}"
)
# 注意：input_variables 中的 "child_summaries_list_str" 对应 HumanMessagePromptTemplate.from_template 中的 {child_summaries_list_str}
parent_human_template_str = (
    "子节点摘要列表:\n```\n{child_summaries_list_str}\n```\n\n" # 将列表转换为字符串以便LLM处理
    "请严格按照以下格式指令进行输出:\n{format_instructions}"
)
PARENT_SUMMARY_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(parent_system_content),
    HumanMessagePromptTemplate.from_template(parent_human_template_str)
]).partial(format_instructions=PARENT_SUMMARY_PARSER.get_format_instructions())


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        
        if settings.HTTP_PROXY:
            logger.info(f"正在使用全局HTTP代理: {settings.HTTP_PROXY}")
            os.environ['HTTP_PROXY'] = settings.HTTP_PROXY
            os.environ['HTTPS_PROXY'] = settings.HTTP_PROXY
            os.environ['NO_PROXY'] = "localhost,127.0.0.1,::1"
            
        logger.info("正在初始化 ModelManager...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 本地模型
        self.embedding_model = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        
        # LLM 提供商实例
        self.chat_provider: LLMProvider = None
        self.summary_provider: LLMProvider = None
        
        self._load_local_models()
        self._configure_llm_providers() # 新增方法
        
        self.initialized = True
        logger.success("ModelManager 初始化成功。")

    def _configure_llm_providers(self):
        """根据配置初始化并配置LLM提供商。"""
        logger.info("正在配置LLM提供商...")
        
        # 配置对话提供商 (Gemini)
        if settings.CHAT_MODEL_PROVIDER.lower() == "gemini":
            if not settings.GOOGLE_API_KEY or "YOUR_GOOGLE" in settings.GOOGLE_API_KEY:
                logger.error("在 .env 文件中未找到或未设置 GOOGLE_API_KEY。")
                sys.exit(1)
            self.chat_provider = GoogleGeminiProvider(
                model_name=settings.CHAT_MODEL_NAME,
                api_key=settings.GOOGLE_API_KEY,
                system_prompt=settings.SYSTEM_PROMPT
            )
        else:
            raise ValueError(f"不支持的对话提供商: {settings.CHAT_MODEL_PROVIDER}")
            
        # 配置摘要提供商 (Volcangine)
        if settings.SUMMARY_MODEL_PROVIDER.lower() == "volcangine":
            if not settings.VOLCANGINE_ARK_API_KEY or "YOUR_VOLCANGINE" in settings.VOLCANGINE_ARK_API_KEY:
                logger.error("在 .env 文件中未找到或未设置 VOLCANGINE_ARK_API_KEY。")
                sys.exit(1)
            self.summary_provider = VolcangineDeepSeekProvider(
                model_name=settings.SUMMARY_MODEL_NAME,
                api_key=settings.VOLCANGINE_ARK_API_KEY
            )
        else:
            raise ValueError(f"不支持的摘要提供商: {settings.SUMMARY_MODEL_PROVIDER}")

    def _load_local_models(self):
        """加载本地的嵌入和重排模型。"""
        logger.info(f"正在加载嵌入模型: {settings.EMBEDDING_MODEL_PATH} on {self.device}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={'device': self.device, 'trust_remote_code': True}
        )
        logger.success("嵌入模型加载成功。")

        logger.info(f"正在加载重排模型: {settings.RERANKER_MODEL_PATH} on {self.device}")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            settings.RERANKER_MODEL_PATH, trust_remote_code=True)
        try:
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                settings.RERANKER_MODEL_PATH,
                torch_dtype="auto", device_map=self.device, attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).eval()
            logger.info("重排模型已加载 (使用 Flash Attention 2)。")
        except (ImportError, Exception):
            logger.warning("Flash Attention 2 不可用，将不使用它来加载重排模型。")
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                settings.RERANKER_MODEL_PATH,
                torch_dtype="auto", device_map=self.device,
                trust_remote_code=True
            ).eval()
        self.reranker_token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
        self.reranker_token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
        logger.success("重排模型加载成功。")

    # --- 委托给LLM提供商的方法 ---

    async def summarize(self, text: str) -> str:
        """(异步) 使用配置好的摘要提供商生成摘要。"""
        if not self.summary_provider:
            logger.error("摘要提供商尚未初始化。")
            return text[:500] + "..." # 备用方案
        try:
            # 这个 summarize 方法使用的是 LLMProvider 内部定义的通用prompt
            return await self.summary_provider.summarize(text)
        except Exception as e:
            logger.error(f"使用 {self.summary_provider.model_name} 进行通用摘要时出错: {e}")
            return text[:500] + "..." # 备用方案

    async def _generate_structured_summary(
        self,
        prompt_template: ChatPromptTemplate,
        parser: PydanticOutputParser,
        input_variables: Dict[str, Any],
        fallback_text: str
    ) -> str:
        """
        内部辅助方法，用于生成结构化摘要并解析。
        """
        if not self.summary_provider:
            logger.error("摘要提供商尚未初始化。无法生成结构化摘要。")
            return fallback_text[:250] + "..." if fallback_text else "摘要生成失败"

        try:
            # 使用 ChatPromptTemplate 格式化消息
            # 注意：partial_variables 已经在模板创建时注入了 format_instructions
            formatted_messages = prompt_template.format_prompt(**input_variables).to_messages()
            
            # 将 Langchain Messages 转换为 LLM Provider期望的字典列表格式
            request_messages = [{"role": msg.type, "content": msg.content} for msg in formatted_messages]

            logger.debug(f"向摘要提供商 {self.summary_provider.model_name} 发送结构化摘要请求...")
            
            # --- 为Gemini适配system message ---
            effective_system_instruction = None
            final_request_messages = request_messages
            
            # 检查是否是GoogleGeminiProvider，并且第一条消息是system role
            # 需要导入 GoogleGeminiProvider: from .llm_providers import GoogleGeminiProvider
            if isinstance(self.summary_provider, GoogleGeminiProvider) and \
               request_messages and request_messages[0].get("role") == "system":
                effective_system_instruction = request_messages[0]["content"]
                final_request_messages = request_messages[1:] # 移除system message，因为它会通过参数传递
                logger.debug("为Gemini适配：将System Message内容移至system_instruction参数。")
            # --- Gemini适配结束 ---

            # 确保使用 summary_provider 的 get_chat_response，而不是 summarize
            llm_response_str = await self.summary_provider.get_chat_response(
                messages=final_request_messages, # 使用适配后的消息
                system_instruction=effective_system_instruction, # 传递适配后的system instruction
                stream=False
            )

            if not llm_response_str or not isinstance(llm_response_str, str):
                logger.error(f"从摘要模型获取的响应为空或类型不正确: {type(llm_response_str)}")
                return fallback_text[:250] + "..." if fallback_text else "摘要内容为空"

            parsed_output = parser.parse(llm_response_str)
            logger.success(f"结构化摘要生成并解析成功。")
            return parsed_output.summary
        except Exception as e:
            logger.error(f"生成或解析结构化摘要时出错 ({self.summary_provider.model_name}): {e}")
            logger.debug(f"发生错误的输入变量: {input_variables}")
            return fallback_text[:250] + "..." if fallback_text else "摘要处理异常"

    async def summarize_leaf_chunk(self, text_chunk: str) -> str:
        """(异步) 使用TDD定义的PROMPT_CREATE_LEAF_SUMMARY为叶子节点文本块生成摘要。"""
        logger.info(f"为叶子节点文本块生成摘要 (长度: {len(text_chunk)})...")
        return await self._generate_structured_summary(
            prompt_template=LEAF_SUMMARY_CHAT_PROMPT,
            parser=LEAF_SUMMARY_PARSER,
            input_variables={"text_chunk": text_chunk},
            fallback_text=text_chunk # 如果失败，返回原始文本块的一部分
        )

    async def summarize_parent_node(self, child_summaries_list: List[str]) -> str:
        """(异步) 使用TDD定义的PROMPT_CREATE_PARENT_SUMMARY为父节点生成摘要。"""
        logger.info(f"基于 {len(child_summaries_list)} 个子摘要生成父节点摘要...")
        # 将子摘要列表转换为单个字符串，每个摘要占一行，以便在提示中清晰显示
        child_summaries_list_str = "\n".join([f"- {s}" for s in child_summaries_list])
        
        # 使用原始的 child_summaries_list （字符串列表）来构建 fallback_text 的基础
        # 如果列表为空，则 fallback_text 也为空字符串，或者一个通用提示
        fallback_content_base = " ".join(child_summaries_list) if child_summaries_list else "子摘要内容缺失"

        return await self._generate_structured_summary(
            prompt_template=PARENT_SUMMARY_CHAT_PROMPT,
            parser=PARENT_SUMMARY_PARSER,
            input_variables={"child_summaries_list_str": child_summaries_list_str},
            fallback_text=fallback_content_base # 如果失败，返回拼接的子摘要的一部分
        )

    async def get_chat_response_async(self, messages: List[Dict[str, str]], system_instruction: str = None, stream: bool = False):
        """
        从配置好的对话提供商获取聊天回复，支持流式传输。
        """
        if not self.chat_provider:
            raise RuntimeError("对话提供商尚未初始化。")

        logger.debug(f">>> 发送请求至对话提供商: {self.chat_provider.__class__.__name__}")
        logger.debug(f"模型: {self.chat_provider.model_name}")
        
        try:
            return await self.chat_provider.get_chat_response(
                messages=messages,
                system_instruction=system_instruction,
                stream=stream
            )
        except Exception as e:
            logger.error(f"从 {self.chat_provider.model_name} 获取聊天回复时出错: {e}")
            raise
    
    # --- 本地模型方法 (保持不变) ---
    
    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the LangChain compatible wrapper.

        Args:
            texts: A list of strings to embed.
            is_query: Set to True if the texts are search queries.

        Returns:
            A list of embeddings.
        """
        if is_query:
            if len(texts) > 1:
                logger.warning("get_embeddings 在 is_query=True 时接收到多个文本，将只嵌入第一个。")
            return [self.embedding_model.embed_query(texts[0])]
        else:
            return self.embedding_model.embed_documents(texts)

    @torch.no_grad()
    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """
        Reranks a list of documents based on a query using the reranker model.

        Args:
            query: The search query.
            docs: A list of document strings to rerank.

        Returns:
            A list of tuples (index, score) where index is the document index and score is the reranking score.
        """
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in docs]
        inputs = self.reranker_tokenizer(
            formatted_pairs, padding=False, truncation='longest_first', return_attention_mask=False,
            max_length=self.reranker_tokenizer.model_max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = prefix_tokens + inputs['input_ids'][i] + suffix_tokens
        inputs = self.reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt").to(self.device)
        batch_scores = self.reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.reranker_token_true_id]
        false_vector = batch_scores[:, self.reranker_token_false_id]
        scores_tensor = torch.stack([false_vector, true_vector], dim=1)
        scores_tensor = torch.nn.functional.log_softmax(scores_tensor, dim=1)
        final_scores = scores_tensor[:, 1].exp().tolist()
        scored_pairs = list(zip(range(len(docs)), final_scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs

# 单例实例，方便全局访问
model_manager = ModelManager()
# The __main__ block is for synchronous testing and will now fail.
# This is expected as the core methods are now async.
# We'll rely on the server for testing.
if __name__ == '__main__':
    print("Skipping direct execution of model_services.py as core methods are now async.") 
