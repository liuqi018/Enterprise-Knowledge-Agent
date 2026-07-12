import json
from typing import Iterator, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session

from AIRAGAgent.agent.enterprise_workflow import EnterpriseKnowledgeWorkflow
from AIRAGAgent.agent.react_agent import ReactAgent
from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.rag.rag_service import RagSummarizeService
from AIRAGAgent.schemas import ChatMessage, ChatResponse
from AIRAGAgent.services.conversation_service import conversation_service
from AIRAGAgent.services.query_classifier import classify_query, needs_clarification
from AIRAGAgent.services.session_service import session_service
from AIRAGAgent.utils.logger_handler import logger


CLARIFICATION_PROMPT = """你是企业知识智能体的澄清助手。用户的问题存在多种可能意图，请生成一段自然、简洁的澄清回复。

要求：
1. 不要回答用户问题本身，只做意图确认。
2. 必须保留下面三个编号选项，编号和含义不能变。
3. 选项内容可以结合用户问题做轻微改写，使其更贴近场景。
4. 语言简洁，不要超过 120 字。

用户问题：{query}

必须包含：
1. 查询相关制度、规则或流程
2. 帮你生成申请草稿或流程方案
3. 做风险校验或合规检查
"""


class ChatService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.agent = None
        self.workflow = None
        self.clarification_chain = PromptTemplate.from_template(CLARIFICATION_PROMPT) | chat_model | StrOutputParser()

    def _agent(self) -> ReactAgent:
        if self.agent is None:
            self.agent = ReactAgent()
        return self.agent

    def _workflow(self) -> EnterpriseKnowledgeWorkflow:
        if self.workflow is None:
            self.workflow = EnterpriseKnowledgeWorkflow()
        return self.workflow

    def answer(
        self,
        query: str,
        session_id: str = None,
        use_agent: bool = True,
        history: List[ChatMessage] = None,
        tenant_id: str = "default",
        user_id: int = 0,
        db: Session = None,
    ) -> ChatResponse:
        sid = self._get_session_id(db, user_id, session_id, query)
        merged_history = history or self._history(db, user_id, sid)
        self._append(db, user_id, sid, "user", query)

        clarification = self._resolve_clarification_choice(merged_history, query)
        route = classify_query(query)
        sources = []
        try:
            if clarification:
                answer, sources = self._answer_clarification_choice(clarification, tenant_id, merged_history)
            elif self._needs_clarification(route):
                answer = self._clarification_answer(query)
            elif route.mode == "chat":
                answer = self._small_talk_answer(query)
            elif use_agent and route.mode == "flow":
                history_payload = self._history_payload(merged_history)
                answer = self._workflow().invoke(query, history_payload).strip()
            else:
                rag_response = self.rag_service.answer(query, tenant_id=tenant_id)
                answer = rag_response.answer
                sources = rag_response.sources
        except Exception as exc:
            logger.error("[chat] failed, fallback to RAG: %s", exc, exc_info=True)
            rag_response = self.rag_service.answer(query, tenant_id=tenant_id)
            answer = rag_response.answer
            sources = rag_response.sources

        self._append(db, user_id, sid, "assistant", answer)
        return ChatResponse(session_id=sid, answer=answer, sources=sources)

    def stream_answer(
        self,
        query: str,
        session_id: str = None,
        use_agent: bool = True,
        history: List[ChatMessage] = None,
        tenant_id: str = "default",
        user_id: int = 0,
        db: Session = None,
    ) -> Iterator[str]:
        sid = self._get_session_id(db, user_id, session_id, query)
        merged_history = history or self._history(db, user_id, sid)
        self._append(db, user_id, sid, "user", query)
        yield self._stream_event("session", {"session_id": sid})

        clarification = self._resolve_clarification_choice(merged_history, query)
        route = classify_query(query)
        full_answer = ""
        sources = []
        try:
            if clarification:
                yield self._stream_event("status", {"message": "已确认处理方式，正在继续处理..."})
                for chunk, chunk_sources in self._stream_clarification_choice(clarification, merged_history, tenant_id):
                    full_answer += chunk
                    sources = chunk_sources
                    yield self._stream_event("delta", {"content": chunk})
            elif self._needs_clarification(route):
                full_answer = self._clarification_answer(query)
                yield self._stream_event("status", {"message": "需要确认你的意图..."})
                yield self._stream_event("delta", {"content": full_answer})
            elif route.mode == "chat":
                full_answer = self._small_talk_answer(query)
                yield self._stream_event("status", {"message": "正在生成回答..."})
                yield self._stream_event("delta", {"content": full_answer})
            elif use_agent and route.mode == "flow":
                yield self._stream_event("status", {"message": "正在生成流程方案..."})
                history_payload = self._history_payload(merged_history)
                for chunk in self._workflow().stream(query, history_payload):
                    if not full_answer:
                        yield self._stream_event("status", {"message": "正在生成回答..."})
                    full_answer += chunk
                    yield self._stream_event("delta", {"content": chunk})
            else:
                yield self._stream_event("status", {"message": "正在快速检索制度库..."})
                for chunk, rag_sources in self.rag_service.stream_answer(query, tenant_id=tenant_id):
                    if not full_answer:
                        yield self._stream_event("status", {"message": "正在生成回答..."})
                    full_answer += chunk
                    sources = rag_sources
                    yield self._stream_event("delta", {"content": chunk})
        except Exception as exc:
            logger.error("[chat stream] failed, fallback to RAG: %s", exc, exc_info=True)
            yield self._stream_event("status", {"message": "出现异常，正在使用 RAG 兜底回答..."})
            rag_response = self.rag_service.answer(query, tenant_id=tenant_id)
            full_answer = rag_response.answer
            sources = rag_response.sources
            yield self._stream_event("delta", {"content": full_answer})

        self._append(db, user_id, sid, "assistant", full_answer.strip())
        yield self._stream_event("done", {"session_id": sid, "sources": sources})

    def _history_payload(self, history: List[ChatMessage]) -> List[dict]:
        return [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in history
        ]

    def _stream_event(self, event: str, data: dict) -> str:
        return json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n"

    def _needs_clarification(self, route) -> bool:
        return needs_clarification(route)

    def _clarification_answer(self, query: str) -> str:
        try:
            answer = self.clarification_chain.invoke({"query": query}).strip()
            if self._valid_clarification_answer(answer):
                return answer
        except Exception as exc:
            logger.warning("[clarification] LLM clarification failed: %s", exc)
        return (
            "我还不确定你想让我按哪种方式处理。请你确认一下：\n"
            "1. 查询相关制度、规则或流程\n"
            "2. 帮你生成申请草稿或流程方案\n"
            "3. 做风险校验或合规检查\n\n"
            "你可以直接回复选项编号，或者补充一句你的目标。"
        )

    def _valid_clarification_answer(self, answer: str) -> bool:
        return all(marker in answer for marker in ["1.", "2.", "3."])

    def _resolve_clarification_choice(self, history: List[ChatMessage], query: str):
        choice = self._parse_clarification_choice(query)
        if not choice:
            return None
        messages = self._history_payload(history or [])
        if not messages:
            return None

        last_assistant_index = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") == "assistant" and self._is_clarification_message(message.get("content", "")):
                last_assistant_index = index
                break
        if last_assistant_index is None:
            return None

        for index in range(last_assistant_index - 1, -1, -1):
            message = messages[index]
            if message.get("role") == "user":
                return {"choice": choice, "original_query": message.get("content", "")}
        return None

    def _is_clarification_message(self, content: str) -> bool:
        return all(marker in content for marker in ["1.", "2.", "3."]) and any(
            keyword in content
            for keyword in ["请回复", "请确认", "哪方面", "哪种方式", "具体需要"]
        )

    def _parse_clarification_choice(self, query: str):
        normalized = query.strip().lower().replace("。", "").replace(".", "")
        if normalized in {"1", "一", "查询", "查制度", "制度", "流程", "规则"}:
            return "policy"
        if normalized in {"2", "二", "生成", "草稿", "申请草稿", "流程方案"}:
            return "flow"
        if normalized in {"3", "三", "风险", "合规", "风险校验", "合规检查"}:
            return "risk"
        return None

    def _answer_clarification_choice(self, clarification: dict, tenant_id: str, history: List[ChatMessage]):
        query = self._clarified_query(clarification)
        if clarification["choice"] == "flow":
            return self._workflow().invoke(query, self._history_payload(history)).strip(), []
        if clarification["choice"] == "risk":
            return self._workflow().invoke(query, self._history_payload(history)).strip(), []
        rag_response = self.rag_service.answer(query, tenant_id=tenant_id)
        return rag_response.answer, rag_response.sources

    def _stream_clarification_choice(self, clarification: dict, history: List[ChatMessage], tenant_id: str):
        query = self._clarified_query(clarification)
        if clarification["choice"] in {"flow", "risk"}:
            for chunk in self._workflow().stream(query, self._history_payload(history)):
                yield str(chunk), []
            return
        for chunk, sources in self.rag_service.stream_answer(query, tenant_id=tenant_id):
            yield chunk, sources

    def _clarified_query(self, clarification: dict) -> str:
        original_query = clarification.get("original_query", "")
        if clarification["choice"] == "policy":
            return f"{original_query} 查询相关制度、规则、流程、审批要求和所需材料"
        if clarification["choice"] == "flow":
            return f"请基于以下需求生成申请草稿或流程方案：{original_query}"
        return f"请对以下事项进行风险校验和合规检查：{original_query}"

    def _small_talk_answer(self, query: str) -> str:
        normalized = query.strip().lower().replace(" ", "")
        if any(pattern in query for pattern in ["没有依据", "未找到依据", "知识库里没有", "知识库没有", "会不会编造", "怎么回答"]):
            return (
                "如果知识库没有检索到明确制度依据，我会直接说明“知识库中没有明确依据”，"
                "不会编造公司制度、审批节点、材料清单或金额标准。必要时我会建议你补充业务场景、制度名称或咨询对应部门。"
            )
        if normalized in {"你是谁", "你能做什么"}:
            return "我是企业知识智能体，可以帮你查询制度、梳理审批流程、生成申请草稿，并支持多轮对话。"
        return "你好，我可以帮你查询企业制度、审批流程、费用标准，也可以生成申请草稿。"

    def _get_session_id(self, db: Session, user_id: int, session_id: str, query: str) -> str:
        if db and user_id:
            return conversation_service.get_or_create_session(db, user_id, session_id, query)
        return session_service.get_or_create(session_id)

    def _history(self, db: Session, user_id: int, session_id: str) -> List[ChatMessage]:
        if db and user_id:
            return conversation_service.history(db, user_id, session_id)
        return session_service.history(session_id)

    def _append(self, db: Session, user_id: int, session_id: str, role: str, content: str) -> None:
        if db and user_id:
            conversation_service.append(db, user_id, session_id, role, content)
            return
        session_service.append(session_id, role, content)


chat_service = ChatService()
