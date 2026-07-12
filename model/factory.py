
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from AIRAGAgent.config.settings import settings
from AIRAGAgent.utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        provider = os.getenv("LLM_PROVIDER", rag_config.get("llm_provider", "dashscope")).lower()
        if provider == "autodl":
            api_key = os.getenv("AUTODL_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("AUTODL_API_KEY is required when LLM_PROVIDER=autodl")

            return ChatOpenAI(
                model=os.getenv("AUTODL_MODEL", rag_config["chat_model_name"]),
                api_key=api_key.strip().strip('"').strip("'"),
                base_url=os.getenv(
                    "AUTODL_BASE_URL",
                    rag_config.get("autodl_base_url", "https://www.autodl.art/api/v1"),
                ),
                streaming=True,
                tiktoken_model_name=os.getenv("AUTODL_TIKTOKEN_MODEL", "gpt-4o"),
            )

        return ChatTongyi(
            model=rag_config["chat_model_name"],
            streaming = True
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])
chat_model=ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()
