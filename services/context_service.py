from __future__ import annotations

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.schemas import ChatMessage
from AIRAGAgent.utils.logger_handler import logger


CONTEXT_REWRITE_PROMPT = """你是企业制度问答系统的上下文改写器。
任务：根据最近对话，把用户当前问题改写成一个可以独立检索知识库的问题。

要求：
1. 只输出改写后的问题，不要解释。
2. 不要回答问题。
3. 不要编造历史中没有的业务对象、金额、部门、时间或条件。
4. 如果当前问题已经完整，不需要改写，则原样输出当前问题。
5. 对“那/这个/上面/刚才/还需要/谁负责/怎么办/需要哪些材料”等追问，要继承最近相关业务主题。

会话摘要：
{summary}

最近对话：
{history}

当前问题：
{query}
"""


CONTEXT_SUMMARY_PROMPT = """你是企业制度问答系统的会话摘要器。
请基于旧摘要和最近对话，更新一个用于后续多轮追问理解的简短摘要。

要求：
1. 只保留与企业制度问答相关的信息。
2. 保留业务主题、金额/时间/部门/人员等条件、用户关注点、已经确认的流程或材料。
3. 不要记录无关闲聊。
4. 不要编造制度结论。
5. 控制在 300 字以内。

旧摘要：
{summary}

最近对话：
{history}

输出新的摘要：
"""


FOLLOW_UP_MARKERS = {
    "那",
    "这个",
    "这些",
    "上述",
    "上面",
    "刚才",
    "继续",
    "还需要",
    "还要",
    "怎么办",
    "怎么处理",
    "谁负责",
    "谁审批",
    "哪些材料",
    "什么材料",
    "多久",
    "提前多久",
    "是否需要",
    "需要吗",
    "呢",
}

BUSINESS_KEYWORDS = {
    "采购",
    "报销",
    "费用",
    "差旅",
    "请假",
    "考勤",
    "加班",
    "入职",
    "离职",
    "转正",
    "薪资",
    "绩效",
    "生产",
    "设备",
    "质量",
    "安全",
    "信息安全",
    "权限",
    "合同",
    "研发",
    "仓库",
}


class ContextService:
    def __init__(self):
        self._summaries: dict[str, str] = {}
        self.rewrite_chain = PromptTemplate.from_template(CONTEXT_REWRITE_PROMPT) | chat_model | StrOutputParser()
        self.summary_chain = PromptTemplate.from_template(CONTEXT_SUMMARY_PROMPT) | chat_model | StrOutputParser()

    def resolve_query(self, session_id: str, query: str, history: List[ChatMessage]) -> str:
        if not settings.CONTEXT_REWRITE_ENABLED:
            return query
        if not history or not self._needs_rewrite(query):
            return query

        summary = self.get_summary(session_id)
        history_text = self._format_history(history, limit=settings.CONTEXT_RECENT_MESSAGES)
        try:
            rewritten = self.rewrite_chain.invoke(
                {"summary": summary or "无", "history": history_text or "无", "query": query}
            ).strip()
            rewritten = self._clean_rewritten_query(rewritten)
            if self._valid_rewrite(query, rewritten):
                logger.info("[Context] query rewritten original=%s rewritten=%s", query, rewritten)
                return rewritten
            logger.info("[Context] ignored invalid rewrite original=%s rewritten=%s", query, rewritten)
        except Exception as exc:
            logger.warning("[Context] rewrite failed, fallback to rule rewrite: %s", exc)
        return self._fallback_rewrite(query, history)

    def update_summary(
        self,
        session_id: str,
        history: List[ChatMessage],
        user_query: str,
        assistant_answer: str,
    ) -> None:
        if not settings.CONTEXT_SUMMARY_ENABLED or not session_id:
            return
        recent = list(history or []) + [
            ChatMessage(role="user", content=user_query),
            ChatMessage(role="assistant", content=assistant_answer[:1200]),
        ]
        if len(recent) < settings.CONTEXT_SUMMARY_TRIGGER_MESSAGES:
            return

        old_summary = self.get_summary(session_id)
        history_text = self._format_history(recent, limit=settings.CONTEXT_RECENT_MESSAGES)
        try:
            summary = self.summary_chain.invoke(
                {"summary": old_summary or "无", "history": history_text or "无"}
            ).strip()
            summary = self._clean_summary(summary)
            if summary:
                self._summaries[session_id] = summary
                logger.info("[Context] summary updated session=%s chars=%s", session_id, len(summary))
                return
        except Exception as exc:
            logger.warning("[Context] summary failed, fallback to lightweight summary: %s", exc)

        fallback = self._fallback_summary(old_summary, user_query, assistant_answer)
        if fallback:
            self._summaries[session_id] = fallback

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def clear(self, session_id: str) -> None:
        self._summaries.pop(session_id, None)

    def _needs_rewrite(self, query: str) -> bool:
        text = query.strip()
        if len(text) <= 4:
            return False
        marker_hit = any(marker in text for marker in FOLLOW_UP_MARKERS)
        business_hit = any(keyword in text for keyword in BUSINESS_KEYWORDS)
        if marker_hit and (len(text) <= 28 or not business_hit):
            return True
        if len(text) <= 14 and not business_hit:
            return True
        return False

    def _fallback_rewrite(self, query: str, history: List[ChatMessage]) -> str:
        previous_user = self._last_user_message(history)
        if previous_user:
            rewritten = f"{previous_user}。追问：{query}"
            logger.info("[Context] fallback rewrite original=%s rewritten=%s", query, rewritten)
            return rewritten
        return query

    def _last_user_message(self, history: List[ChatMessage]) -> str:
        for message in reversed(history or []):
            if message.role == "user" and message.content.strip():
                return message.content.strip()
        return ""

    def _format_history(self, history: List[ChatMessage], limit: int) -> str:
        rows = []
        for message in (history or [])[-limit:]:
            role = "用户" if message.role == "user" else "助手"
            content = " ".join(message.content.split())
            rows.append(f"{role}: {content[:500]}")
        return "\n".join(rows)

    def _clean_rewritten_query(self, text: str) -> str:
        cleaned = text.strip().strip("`").strip()
        for prefix in ["改写后：", "改写后的问题：", "独立问题：", "问题："]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip("“”\"'")

    def _valid_rewrite(self, original: str, rewritten: str) -> bool:
        if not rewritten:
            return False
        if len(rewritten) > 160:
            return False
        if any(bad in rewritten for bad in ["无法改写", "不能确定", "不知道"]):
            return False
        return original in rewritten or len(rewritten) >= len(original)

    def _clean_summary(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if cleaned.startswith("新的摘要："):
            cleaned = cleaned[len("新的摘要：") :].strip()
        return cleaned[:500]

    def _fallback_summary(self, old_summary: str, user_query: str, assistant_answer: str) -> str:
        item = f"最近主题：{user_query.strip()}；系统已回答：{assistant_answer.strip()[:160]}"
        if old_summary:
            return f"{old_summary} {item}"[:500]
        return item[:500]


context_service = ContextService()
