from typing import List, Optional, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session

from AIRAGAgent.config.settings import settings
from AIRAGAgent.db.models import Conversation, Message
from AIRAGAgent.schemas import ChatMessage


MESSAGE_STATUS_SUCCESS = "success"
MESSAGE_STATUS_PENDING = "pending"
MESSAGE_STATUS_FAILED = "failed"
FAILED_MESSAGE_CONTENT = "\u751f\u6210\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002"


class ConversationService:
    def get_or_create_session(self, db: Session, user_id: int, session_id: Optional[str], first_query: str = "") -> str:
        session_id, _ = self.get_or_create_session_info(db, user_id, session_id, first_query)
        return session_id

    def get_or_create_session_info(
        self,
        db: Session,
        user_id: int,
        session_id: Optional[str],
        first_query: str = "",
    ) -> Tuple[str, bool]:
        if session_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
                .first()
            )
            if conversation:
                return conversation.session_id, False

        new_session_id = str(uuid4())
        title = first_query[:30] if first_query else "\u65b0\u4f1a\u8bdd"
        db.add(Conversation(session_id=new_session_id, user_id=user_id, title=title))
        db.commit()
        return new_session_id, True

    def history(self, db: Session, user_id: int, session_id: str) -> List[ChatMessage]:
        rows = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.user_id == user_id)
            .filter(Message.status == MESSAGE_STATUS_SUCCESS)
            .order_by(Message.id.desc())
            .limit(settings.MAX_HISTORY_MESSAGES)
            .all()
        )
        return [ChatMessage(role=row.role, content=row.content) for row in reversed(rows)]

    def append(self, db: Session, user_id: int, session_id: str, role: str, content: str) -> None:
        self.append_message(db, user_id, session_id, role, content)

    def append_message(
        self,
        db: Session,
        user_id: int,
        session_id: str,
        role: str,
        content: str,
        status: str = MESSAGE_STATUS_SUCCESS,
        retryable: bool = False,
        parent_message_id: int = None,
        error_message: str = None,
    ) -> Message:
        message = Message(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            status=status,
            retryable=retryable,
            parent_message_id=parent_message_id,
            error_message=error_message,
        )
        db.add(message)
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
            .first()
        )
        if conversation and role == "user" and conversation.title == "\u65b0\u4f1a\u8bdd":
            conversation.title = content[:30]
        db.commit()
        db.refresh(message)
        return message

    def create_pending_assistant(
        self,
        db: Session,
        user_id: int,
        session_id: str,
        parent_message_id: int,
    ) -> Message:
        return self.append_message(
            db,
            user_id,
            session_id,
            "assistant",
            "",
            status=MESSAGE_STATUS_PENDING,
            retryable=False,
            parent_message_id=parent_message_id,
        )

    def mark_message_success(self, db: Session, message_id: int, user_id: int, content: str) -> Optional[Message]:
        message = self.get_message(db, message_id, user_id)
        if not message:
            return None
        message.content = content or ""
        message.status = MESSAGE_STATUS_SUCCESS
        message.error_message = None
        message.retryable = False
        db.commit()
        db.refresh(message)
        return message

    def mark_message_failed(
        self,
        db: Session,
        message_id: int,
        user_id: int,
        error_message: str,
        content: str = FAILED_MESSAGE_CONTENT,
        retryable: bool = True,
    ) -> Optional[Message]:
        message = self.get_message(db, message_id, user_id)
        if not message:
            return None
        message.content = content
        message.status = MESSAGE_STATUS_FAILED
        message.error_message = (error_message or "")[:2000]
        message.retryable = retryable
        db.commit()
        db.refresh(message)
        return message

    def reset_assistant_for_retry(self, db: Session, message_id: int, user_id: int) -> Optional[Message]:
        message = self.get_message(db, message_id, user_id)
        if not message or message.role != "assistant":
            return None
        message.content = ""
        message.status = MESSAGE_STATUS_PENDING
        message.error_message = None
        message.retryable = False
        db.commit()
        db.refresh(message)
        return message

    def list_conversations(self, db: Session, user_id: int) -> List[Conversation]:
        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def messages(self, db: Session, user_id: int, session_id: str) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.user_id == user_id)
            .order_by(Message.id.asc())
            .all()
        )

    def get_message(self, db: Session, message_id: int, user_id: int) -> Optional[Message]:
        return db.query(Message).filter(Message.id == message_id, Message.user_id == user_id).first()

    def retry_context(self, db: Session, assistant_message_id: int, user_id: int) -> Tuple[Optional[Message], Optional[Message]]:
        assistant = self.get_message(db, assistant_message_id, user_id)
        if not assistant or assistant.role != "assistant" or assistant.status != MESSAGE_STATUS_FAILED:
            return None, None
        if not assistant.retryable or not assistant.parent_message_id:
            return assistant, None
        user_message = self.get_message(db, assistant.parent_message_id, user_id)
        if not user_message or user_message.role != "user":
            return assistant, None
        return assistant, user_message

    def get_summary(self, db: Session, user_id: int, session_id: str) -> str:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
            .first()
        )
        return (conversation.summary or "") if conversation else ""

    def update_summary(self, db: Session, user_id: int, session_id: str, summary: str) -> None:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
            .first()
        )
        if not conversation:
            return
        conversation.summary = summary or ""
        db.commit()

    def clear_summary(self, db: Session, user_id: int, session_id: str) -> None:
        self.update_summary(db, user_id, session_id, "")

    def clear(self, db: Session, user_id: int, session_id: str) -> bool:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
            .first()
        )
        if not conversation:
            return False
        db.query(Message).filter(Message.session_id == session_id, Message.user_id == user_id).delete()
        db.delete(conversation)
        db.commit()
        return True


conversation_service = ConversationService()
