from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    success: bool = True
    data: Any = None
    message: str = "ok"


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_agent: bool = True
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    assistant_message_id: Optional[int] = None


class ConversationResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    status: str = "success"
    retryable: bool = False
    error_message: Optional[str] = None
    parent_message_id: Optional[int] = None
    created_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_id: str = "default"


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    tenant_id: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    tenant_id: str
    role: str


class KnowledgeIngestRequest(BaseModel):
    force: bool = False


class KnowledgeIngestResponse(BaseModel):
    backend: str
    scanned_files: int
    indexed_files: int
    skipped_files: int
    indexed_chunks: int
    deleted_files: int = 0
    deleted_chunks: int = 0
    task_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class KnowledgeTaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[KnowledgeIngestResponse] = None
    error: Optional[str] = None


class KnowledgeStatsResponse(BaseModel):
    backend: str
    document_count: int
    data_path: str
    manifest_path: str


class RagRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class RagResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
