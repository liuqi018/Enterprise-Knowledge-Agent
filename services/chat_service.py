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
from AIRAGAgent.services.access_control_service import access_control_service
from AIRAGAgent.services.conversation_service import FAILED_MESSAGE_CONTENT, conversation_service
from AIRAGAgent.services.context_service import context_service
from AIRAGAgent.services.query_classifier import classify_query, needs_clarification
from AIRAGAgent.services.session_service import session_service
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.trace import elapsed_ms, log_trace, new_trace_id, now_ms, short_text, trace_scope


CLARIFICATION_PROMPT = """你是企业知识智能体的澄清助手。用户的问题存在多种可能处理方式，请根据问题动态生成澄清菜单。

要求：
1. 不要回答用户问题本身，只做意图确认。
2. 必须输出 1、2、3 三个编号选项，用户可以回复编号继续。
3. 三个选项要结合用户问题里的业务对象改写，不要总是使用固定套话。
4. 三个选项的方向分别是：查询制度依据、生成流程/申请草稿、做风险/合规检查。
5. 语言简洁，不要超过 140 字。

用户问题：{query}

输出格式：
你想让我按哪种方式处理“业务对象”？
1. ...
2. ...
3. ...
"""

CHAT_PROMPT = """你是企业知识智能体的对话助手。请回答用户的普通聊天、系统能力或回答规范类问题。

边界：
1. 不要检索或引用企业制度资料。
2. 不要声称“我查到了某制度”或输出来源。
3. 如果用户问没有依据时如何回答，要说明：没有明确制度依据时会直接说明，不会编造公司制度、审批节点、材料清单、金额或天数。
4. 如果用户询问具体制度、流程、材料、审批标准，只说明可以帮他查询，并建议直接提出具体问题；不要替制度库作答。
5. 使用中文，回答自然、简洁，控制在 150 字以内。

用户问题：{query}
"""


class ChatService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.agent = None
        self.workflow = None
        self.clarification_chain = PromptTemplate.from_template(CLARIFICATION_PROMPT) | chat_model | StrOutputParser()
        self.chat_chain = PromptTemplate.from_template(CHAT_PROMPT) | chat_model | StrOutputParser()

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
        user_role: str = "user",
        db: Session = None,
        trace_id: str = None,
    ) -> ChatResponse:
        with trace_scope(trace_id or new_trace_id()):
            total_start = now_ms()
            sid, is_new_session = self._get_session_context(db, user_id, session_id, query)
            merged_history = [] if is_new_session else (history or self._history(db, user_id, sid))
            log_trace(
                logger,
                "chat_start",
                mode="sync",
                session_id=sid,
                requested_session_id=session_id,
                is_new_session=is_new_session,
                tenant_id=tenant_id,
                user_role=user_role,
                history_count=len(merged_history or []),
                query=short_text(query),
            )
            user_message = self._append(db, user_id, sid, "user", query)
            assistant_message = self._create_pending_assistant(db, user_id, sid, user_message)

            clarification = self._resolve_clarification_choice(merged_history, query)
            summary = self._summary(db, user_id, sid)
            contextual_follow_up = (
                not clarification and context_service.is_contextual_follow_up(query, merged_history)
            )
            effective_query = query if clarification else context_service.resolve_query(sid, query, merged_history, summary)
            log_trace(
                logger,
                "context_resolved",
                session_id=sid,
                contextual_follow_up=contextual_follow_up,
                clarification=bool(clarification),
                summary_chars=len(summary or ""),
                effective_changed=effective_query != query,
                effective_query=short_text(effective_query),
            )
            access_decision = access_control_service.can_access_query(effective_query, user_role)
            route = classify_query(effective_query)
            log_trace(
                logger,
                "route_decision",
                mode=route.mode,
                retrieval_mode=route.retrieval_mode,
                confidence=route.confidence,
                access_allowed=access_decision.allowed,
                access_domain=access_decision.domain,
                access_action=access_decision.action,
                access_reason=access_decision.reason,
                reason=route.reason,
            )
            sources = []
            branch = "unknown"
            try:
                if not access_decision.allowed:
                    branch = "access_denied"
                    answer = access_decision.message
                elif clarification:
                    branch = f"clarification_{clarification['choice']}"
                    answer, sources = self._answer_clarification_choice(clarification, tenant_id, merged_history)
                elif self._needs_clarification(route) and not contextual_follow_up:
                    branch = "clarification_menu"
                    answer = self._clarification_answer(effective_query)
                elif route.mode == "chat":
                    branch = "chat"
                    answer = self._small_talk_answer(effective_query)
                elif use_agent and route.mode == "flow":
                    branch = "workflow"
                    history_payload = self._history_payload(merged_history)
                    answer = self._workflow().invoke(effective_query, history_payload).strip()
                else:
                    branch = "rag"
                    rag_response = self.rag_service.answer(effective_query, tenant_id=tenant_id)
                    answer = rag_response.answer
                    sources = rag_response.sources
            except Exception as exc:
                branch = "rag_fallback"
                logger.error("[chat] failed, fallback to RAG: %s", exc, exc_info=True)
                log_trace(logger, "chat_error_fallback", error=f"{type(exc).__name__}: {exc}")
                try:
                    rag_response = self.rag_service.answer(effective_query, tenant_id=tenant_id)
                    answer = rag_response.answer
                    sources = rag_response.sources
                except Exception as fallback_exc:
                    self._mark_assistant_failed(db, user_id, assistant_message, fallback_exc)
                    raise

            self._mark_assistant_success(db, user_id, assistant_message, answer)
            new_summary = context_service.update_summary(sid, merged_history, query, answer, old_summary=summary)
            self._update_summary(db, user_id, sid, new_summary)
            log_trace(
                logger,
                "chat_done",
                branch=branch,
                elapsed_ms=elapsed_ms(total_start),
                answer_chars=len(answer or ""),
                sources_count=len(sources or []),
            )
            return ChatResponse(
                session_id=sid,
                answer=answer,
                sources=sources,
                assistant_message_id=getattr(assistant_message, "id", None),
            )

    def stream_answer(
        self,
        query: str,
        session_id: str = None,
        use_agent: bool = True,
        history: List[ChatMessage] = None,
        tenant_id: str = "default",
        user_id: int = 0,
        user_role: str = "user",
        db: Session = None,
        trace_id: str = None,
    ) -> Iterator[str]:
        active_trace_id = trace_id or new_trace_id()
        with trace_scope(active_trace_id):
            total_start = now_ms()
            sid, is_new_session = self._get_session_context(db, user_id, session_id, query)
            merged_history = [] if is_new_session else (history or self._history(db, user_id, sid))
            log_trace(
                logger,
                "chat_start",
                trace_id=active_trace_id,
                mode="stream",
                session_id=sid,
                requested_session_id=session_id,
                is_new_session=is_new_session,
                tenant_id=tenant_id,
                user_role=user_role,
                history_count=len(merged_history or []),
                query=short_text(query),
            )
            user_message = self._append(db, user_id, sid, "user", query)
            assistant_message = self._create_pending_assistant(db, user_id, sid, user_message)
            yield self._stream_event(
                "session",
                {
                    "session_id": sid,
                    "assistant_message_id": getattr(assistant_message, "id", None),
                },
            )

            clarification = self._resolve_clarification_choice(merged_history, query)
            summary = self._summary(db, user_id, sid)
            contextual_follow_up = (
                not clarification and context_service.is_contextual_follow_up(query, merged_history)
            )
            effective_query = query if clarification else context_service.resolve_query(sid, query, merged_history, summary)
            log_trace(
                logger,
                "context_resolved",
                trace_id=active_trace_id,
                session_id=sid,
                contextual_follow_up=contextual_follow_up,
                clarification=bool(clarification),
                summary_chars=len(summary or ""),
                effective_changed=effective_query != query,
                effective_query=short_text(effective_query),
            )
            access_decision = access_control_service.can_access_query(effective_query, user_role)
            route = classify_query(effective_query)
            log_trace(
                logger,
                "route_decision",
                trace_id=active_trace_id,
                mode=route.mode,
                retrieval_mode=route.retrieval_mode,
                confidence=route.confidence,
                access_allowed=access_decision.allowed,
                access_domain=access_decision.domain,
                access_action=access_decision.action,
                access_reason=access_decision.reason,
                reason=route.reason,
            )
            full_answer = ""
            sources = []
            branch = "unknown"
            try:
                if not access_decision.allowed:
                    branch = "access_denied"
                    full_answer = access_decision.message
                    yield self._stream_event("status", {"message": "\u5f53\u524d\u95ee\u9898\u6d89\u53ca\u654f\u611f\u5236\u5ea6\u6743\u9650..."})
                    yield self._stream_event("delta", {"content": full_answer})
                elif clarification:
                    branch = f"clarification_{clarification['choice']}"
                    yield self._stream_event("status", {"message": "\u5df2\u786e\u8ba4\u5904\u7406\u65b9\u5f0f\uff0c\u6b63\u5728\u7ee7\u7eed\u5904\u7406..."})
                    for chunk, chunk_sources in self._stream_clarification_choice(clarification, merged_history, tenant_id):
                        full_answer += chunk
                        sources = chunk_sources
                        yield self._stream_event("delta", {"content": chunk})
                elif self._needs_clarification(route) and not contextual_follow_up:
                    branch = "clarification_menu"
                    full_answer = self._clarification_answer(effective_query)
                    yield self._stream_event("status", {"message": "\u9700\u8981\u786e\u8ba4\u4f60\u7684\u610f\u56fe..."})
                    yield self._stream_event("delta", {"content": full_answer})
                elif route.mode == "chat":
                    branch = "chat"
                    full_answer = self._small_talk_answer(effective_query)
                    yield self._stream_event("status", {"message": "\u6b63\u5728\u751f\u6210\u56de\u7b54..."})
                    yield self._stream_event("delta", {"content": full_answer})
                elif use_agent and route.mode == "flow":
                    branch = "workflow"
                    yield self._stream_event("status", {"message": "\u6b63\u5728\u751f\u6210\u6d41\u7a0b\u65b9\u6848..."})
                    history_payload = self._history_payload(merged_history)
                    for chunk in self._workflow().stream(effective_query, history_payload):
                        if not full_answer:
                            yield self._stream_event("status", {"message": "\u6b63\u5728\u751f\u6210\u56de\u7b54..."})
                        full_answer += chunk
                        yield self._stream_event("delta", {"content": chunk})
                else:
                    branch = "rag"
                    yield self._stream_event("status", {"message": "\u6b63\u5728\u5feb\u901f\u68c0\u7d22\u5236\u5ea6\u5e93..."})
                    for chunk, rag_sources in self.rag_service.stream_answer(effective_query, tenant_id=tenant_id):
                        if not full_answer:
                            yield self._stream_event("status", {"message": "\u6b63\u5728\u751f\u6210\u56de\u7b54..."})
                        full_answer += chunk
                        sources = rag_sources
                        yield self._stream_event("delta", {"content": chunk})
            except Exception as exc:
                branch = "rag_fallback"
                logger.error("[chat stream] failed, fallback to RAG: %s", exc, exc_info=True)
                log_trace(logger, "chat_stream_error_fallback", trace_id=active_trace_id, error=f"{type(exc).__name__}: {exc}")
                yield self._stream_event("status", {"message": "\u51fa\u73b0\u5f02\u5e38\uff0c\u6b63\u5728\u4f7f\u7528 RAG \u515c\u5e95\u56de\u7b54..."})
                try:
                    rag_response = self.rag_service.answer(effective_query, tenant_id=tenant_id)
                    full_answer = rag_response.answer
                    sources = rag_response.sources
                    yield self._stream_event("delta", {"content": full_answer})
                except Exception as fallback_exc:
                    error_text = f"{type(fallback_exc).__name__}: {fallback_exc}"
                    self._mark_assistant_failed(db, user_id, assistant_message, fallback_exc)
                    log_trace(logger, "chat_stream_failed", trace_id=active_trace_id, error=error_text)
                    yield self._stream_event(
                        "error",
                        {
                            "message": FAILED_MESSAGE_CONTENT,
                            "assistant_message_id": getattr(assistant_message, "id", None),
                            "retryable": True,
                        },
                    )
                    return

            self._mark_assistant_success(db, user_id, assistant_message, full_answer.strip())
            new_summary = context_service.update_summary(
                sid,
                merged_history,
                query,
                full_answer.strip(),
                old_summary=summary,
            )
            self._update_summary(db, user_id, sid, new_summary)
            log_trace(
                logger,
                "chat_done",
                trace_id=active_trace_id,
                branch=branch,
                elapsed_ms=elapsed_ms(total_start),
                answer_chars=len(full_answer.strip()),
                sources_count=len(sources or []),
            )
            yield self._stream_event(
                "done",
                {
                    "session_id": sid,
                    "sources": sources,
                    "assistant_message_id": getattr(assistant_message, "id", None),
                },
            )

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
        topic = self._clarification_topic(query)
        return (
            f"你想让我按哪种方式处理“{topic}”？\n"
            f"1. 查询“{topic}”相关制度依据、办理规则或流程\n"
            f"2. 基于“{topic}”生成流程方案、申请说明或草稿\n"
            f"3. 对“{topic}”做风险校验、合规检查或注意事项梳理\n\n"
            "你可以直接回复选项编号，或者补充一句你的目标。"
        )

    def _valid_clarification_answer(self, answer: str) -> bool:
        return all(marker in answer for marker in ["1.", "2.", "3."]) and len(answer.strip()) <= 220

    def _clarification_topic(self, query: str) -> str:
        topic = " ".join(query.strip().split())
        for prefix in ["请问", "我想", "帮我", "需要", "查询", "生成", "做一下"]:
            topic = topic.replace(prefix, "")
        topic = topic.strip("，。？！；:： ")
        return topic[:28] or "这个事项"

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
        try:
            answer = self.chat_chain.invoke({"query": query}).strip()
            if answer:
                return answer
        except Exception as exc:
            logger.warning("[chat] LLM small talk failed: %s", exc)
        return "我是企业知识智能体，可以帮你查询制度、梳理审批流程、生成申请草稿。没有明确制度依据时，我会直接说明，不会编造。"


    def retry_answer(
        self,
        assistant_message_id: int,
        tenant_id: str = "default",
        user_id: int = 0,
        user_role: str = "user",
        db: Session = None,
        use_agent: bool = True,
        trace_id: str = None,
    ) -> ChatResponse:
        if not db or not user_id:
            raise ValueError("retry requires a persisted user session")

        assistant_message, user_message = conversation_service.retry_context(db, assistant_message_id, user_id)
        if not assistant_message or not user_message:
            raise ValueError("message is not retryable")

        assistant_message = conversation_service.reset_assistant_for_retry(db, assistant_message.id, user_id)
        query = user_message.content
        sid = user_message.session_id

        with trace_scope(trace_id or new_trace_id()):
            total_start = now_ms()
            merged_history = conversation_service.history(db, user_id, sid)
            log_trace(
                logger,
                "chat_retry_start",
                session_id=sid,
                tenant_id=tenant_id,
                user_role=user_role,
                assistant_message_id=assistant_message.id,
                history_count=len(merged_history or []),
                query=short_text(query),
            )
            sources = []
            branch = "unknown"
            try:
                clarification = self._resolve_clarification_choice(merged_history, query)
                summary = self._summary(db, user_id, sid)
                contextual_follow_up = (
                    not clarification and context_service.is_contextual_follow_up(query, merged_history)
                )
                effective_query = query if clarification else context_service.resolve_query(sid, query, merged_history, summary)
                access_decision = access_control_service.can_access_query(effective_query, user_role)
                route = classify_query(effective_query)
                log_trace(
                    logger,
                    "route_decision",
                    mode=route.mode,
                    retrieval_mode=route.retrieval_mode,
                    confidence=route.confidence,
                    access_allowed=access_decision.allowed,
                    access_domain=access_decision.domain,
                    access_action=access_decision.action,
                    access_reason=access_decision.reason,
                    reason=route.reason,
                )
                try:
                    if not access_decision.allowed:
                        branch = "access_denied"
                        answer = access_decision.message
                    elif clarification:
                        branch = f"clarification_{clarification['choice']}"
                        answer, sources = self._answer_clarification_choice(clarification, tenant_id, merged_history)
                    elif self._needs_clarification(route) and not contextual_follow_up:
                        branch = "clarification_menu"
                        answer = self._clarification_answer(effective_query)
                    elif route.mode == "chat":
                        branch = "chat"
                        answer = self._small_talk_answer(effective_query)
                    elif use_agent and route.mode == "flow":
                        branch = "workflow"
                        history_payload = self._history_payload(merged_history)
                        answer = self._workflow().invoke(effective_query, history_payload).strip()
                    else:
                        branch = "rag"
                        rag_response = self.rag_service.answer(effective_query, tenant_id=tenant_id)
                        answer = rag_response.answer
                        sources = rag_response.sources
                except Exception as exc:
                    branch = "rag_fallback"
                    logger.error("[chat retry] failed, fallback to RAG: %s", exc, exc_info=True)
                    log_trace(logger, "chat_retry_error_fallback", error=f"{type(exc).__name__}: {exc}")
                    rag_response = self.rag_service.answer(effective_query, tenant_id=tenant_id)
                    answer = rag_response.answer
                    sources = rag_response.sources

                self._mark_assistant_success(db, user_id, assistant_message, answer)
                new_summary = context_service.update_summary(sid, merged_history, query, answer, old_summary=summary)
                self._update_summary(db, user_id, sid, new_summary)
                log_trace(
                    logger,
                    "chat_retry_done",
                    branch=branch,
                    elapsed_ms=elapsed_ms(total_start),
                    answer_chars=len(answer or ""),
                    sources_count=len(sources or []),
                )
                return ChatResponse(
                    session_id=sid,
                    answer=answer,
                    sources=sources,
                    assistant_message_id=assistant_message.id,
                )
            except Exception as exc:
                self._mark_assistant_failed(db, user_id, assistant_message, exc)
                log_trace(logger, "chat_retry_failed", error=f"{type(exc).__name__}: {exc}")
                raise

    def _create_pending_assistant(self, db: Session, user_id: int, session_id: str, user_message):
        if db and user_id and user_message:
            return conversation_service.create_pending_assistant(db, user_id, session_id, user_message.id)
        return None

    def _mark_assistant_success(self, db: Session, user_id: int, assistant_message, answer: str) -> None:
        if db and user_id and assistant_message:
            conversation_service.mark_message_success(db, assistant_message.id, user_id, answer)

    def _mark_assistant_failed(self, db: Session, user_id: int, assistant_message, exc: Exception) -> None:
        if db and user_id and assistant_message:
            conversation_service.mark_message_failed(
                db,
                assistant_message.id,
                user_id,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _get_session_id(self, db: Session, user_id: int, session_id: str, query: str) -> str:
        return self._get_session_context(db, user_id, session_id, query)[0]

    def _get_session_context(self, db: Session, user_id: int, session_id: str, query: str):
        if db and user_id:
            return conversation_service.get_or_create_session_info(db, user_id, session_id, query)
        sid = session_service.get_or_create(session_id)
        return sid, not bool(session_id)

    def _history(self, db: Session, user_id: int, session_id: str) -> List[ChatMessage]:
        if db and user_id:
            return conversation_service.history(db, user_id, session_id)
        return session_service.history(session_id)

    def _append(self, db: Session, user_id: int, session_id: str, role: str, content: str):
        if db and user_id:
            return conversation_service.append_message(db, user_id, session_id, role, content)
        session_service.append(session_id, role, content)
        return None

    def _summary(self, db: Session, user_id: int, session_id: str) -> str:
        if db and user_id:
            return conversation_service.get_summary(db, user_id, session_id)
        return context_service.get_summary(session_id)

    def _update_summary(self, db: Session, user_id: int, session_id: str, summary: str) -> None:
        if not summary:
            return
        if db and user_id:
            conversation_service.update_summary(db, user_id, session_id, summary)

    def clear_context(self, session_id: str) -> None:
        context_service.clear(session_id)


chat_service = ChatService()
