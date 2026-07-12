from typing import Dict, List, Optional
from uuid import uuid4

from AIRAGAgent.config.settings import settings
from AIRAGAgent.schemas import ChatMessage


class SessionService:
    def __init__(self):
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        new_session_id = session_id or str(uuid4())
        self._sessions.setdefault(new_session_id, [])
        return new_session_id

    def history(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.get(session_id, [])[-settings.MAX_HISTORY_MESSAGES :]

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(ChatMessage(role=role, content=content))
        self._sessions[session_id] = self._sessions[session_id][-settings.MAX_HISTORY_MESSAGES :]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


session_service = SessionService()
