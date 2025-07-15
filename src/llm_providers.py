# src/llm_providers.py
import abc
import os
import sys
from typing import List, Dict, Optional, AsyncGenerator
from loguru import logger

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入特定厂商的SDK
import google.generativeai as genai
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkAPIStatusError, ArkRateLimitError
import asyncio
from .config import settings

# --- 所有LLM提供商的抽象基类 ---

class LLMProvider(abc.ABC):
    """
    一个抽象基类，为所有LLM提供商定义了标准接口。
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abc.abstractmethod
    async def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """
        生成聊天回复。对于非流式请求，返回字符串；对于流式请求，返回异步生成器。
        """
        pass

    async def summarize(self, text: str) -> str:
        """
        为给定文本生成摘要。
        这是一个可以使用聊天端点实现的默认方法。
        """
        logger.info(f"正在使用模型 {self.model_name} 为长度为 {len(text)} 的文本生成摘要...")
        prompt = f'''请为以下文本生成一个精炼的核心摘要。

要求：
1.  **高度浓缩**：摘要必须显著短于原文。
2.  **保留关键**：只包含最重要的信息，即"是什么"和"如何工作"。
3.  **忠于原文**：不要添加任何原文未提及的观点、解释或信息。

{text}'''
        
        # 我们期望摘要功能返回的是非流式的完整结果
        response_generator = await self.get_chat_response(
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        # 因为 stream=False，这里会得到一个字符串
        summary = response_generator
        
        if len(summary) > len(text):
            logger.warning(f"生成的摘要比原文还长，将返回原文。")
            return text
            
        logger.success(f"摘要生成成功，长度为: {len(summary)}。")
        return summary

# --- 针对 Google Gemini 的具体实现 ---

class GoogleGeminiProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str, system_prompt: str):
        super().__init__(model_name, api_key)
        self.default_system_prompt = system_prompt
        try:
            genai.configure(api_key=self.api_key)
            logger.success(f"Google Gemini 提供商配置成功，模型: {self.model_name}")
        except Exception as e:
            logger.error(f"配置 Google Gemini 失败: {e}")
            raise

    async def _stream_text_from_response(self, response_generator):
        """从流式响应中提取文本块"""
        async for chunk in response_generator:
            try:
                yield chunk.text
            except ValueError:
                logger.debug("跳过一个不含文本的 Gemini 流式数据块。")
                continue

    async def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        
        final_system_instruction = system_instruction or self.default_system_prompt
        
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=final_system_instruction
        )
        
        gemini_messages = [
            {"role": msg["role"] if msg["role"] != "assistant" else "model", "parts": [msg["content"]]}
            for msg in messages
            if msg["role"] in ["user", "assistant"]
        ]
        
        try:
            response = await model.generate_content_async(gemini_messages, stream=stream)
            if stream:
                return self._stream_text_from_response(response)
            else:
                return response.text
        except Exception as e:
            logger.error(f"调用 Gemini API 出错: {e}")
            raise

# --- 针对火山引擎 DeepSeek 的具体实现 (使用Ark SDK) ---

class VolcangineDeepSeekProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        if not api_key or "YOUR_VOLCANGINE" in api_key:
            logger.error("在 .env 文件中未找到或未设置 VOLCANGINE_ARK_API_KEY。")
            raise ValueError("Volcangine API Key 未配置")
            
        self.client = Ark(api_key=self.api_key)
        logger.success(f"火山引擎 Ark 提供商配置成功，模型: {self.model_name}")

    async def _stream_text_from_response(self, response_generator):
        """从流式响应中提取文本块 (异步处理同步的SDK)"""
        # 这个生成器本身是同步的，我们在一个异步方法中迭代它
        # 这种方式在 to_thread 内部工作良好
        for chunk in response_generator:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        
        req_messages = []
        if system_instruction:
            req_messages.append({"role": "system", "content": system_instruction})
        req_messages.extend(messages)

        last_exception = None
        for attempt in range(settings.LLM_RETRY_ATTEMPTS):
            try:
                if stream:
                    response_stream = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model_name,
                        messages=req_messages,
                        stream=True
                    )
                    return self._stream_text_from_response(response_stream)
                else:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model_name,
                        messages=req_messages,
                        stream=False
                    )
                    return response.choices[0].message.content
            except ArkRateLimitError as e:
                last_exception = e
                delay = settings.LLM_INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"调用火山引擎API时服务器过载 (429)。"
                    f"将在 {delay:.1f} 秒后进行第 {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS} 次重试..."
                )
                await asyncio.sleep(delay)
                continue
            except ArkAPIStatusError as e:
                logger.error(f"调用火山引擎 Ark API 时发生非429的API状态错误: {e}")
                raise e
            except Exception as e:
                logger.error(f"调用火山引擎 Ark API 时发生未知错误: {e}")
                raise e
        
        # 如果所有重试都失败了，则抛出最后一次捕获的异常
        raise last_exception 