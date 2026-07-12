from typing import List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from AIRAGAgent.config.settings import settings
from AIRAGAgent.db.models import Conversation, Message
from AIRAGAgent.schemas import ChatMessage


class ConversationService:
    def get_or_create_session(self, db: Session, user_id: int, session_id: Optional[str], first_query: str = "") -> str:
        if session_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
                .first()
            )
            if conversation:
                return conversation.session_id

        new_session_id = str(uuid4())
        title = first_query[:30] if first_query else "新会话"
        db.add(Conversation(session_id=new_session_id, user_id=user_id, title=title))
        db.commit()
        return new_session_id

    def history(self, db: Session, user_id: int, session_id: str) -> List[ChatMessage]:
        rows = (
            db.query(Message)
            .filter(Message.session_id == session_id, Message.user_id == user_id)
            .order_by(Message.id.desc())
            .limit(settings.MAX_HISTORY_MESSAGES)
            .all()
        )
        return [ChatMessage(role=row.role, content=row.content) for row in reversed(rows)]

    def append(self, db: Session, user_id: int, session_id: str, role: str, content: str) -> None:
        db.add(Message(session_id=session_id, user_id=user_id, role=role, content=content))
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id, Conversation.user_id == user_id)
            .first()
        )
        if conversation and role == "user" and conversation.title == "新会话":
            conversation.title = content[:30]
        db.commit()

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
