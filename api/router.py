from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from AIRAGAgent.api.deps import get_current_user
from AIRAGAgent.db.models import User
from AIRAGAgent.db.session import get_db
from AIRAGAgent.knowledge.service import KnowledgeBaseService
from AIRAGAgent.rag.rag_service import RagSummarizeService
from AIRAGAgent.schemas import ApiResponse, ChatRequest, ConversationResponse, KnowledgeIngestRequest, LoginRequest, LoginResponse, MessageResponse, RagRequest, RegisterRequest, UserResponse
from AIRAGAgent.services.chat_service import chat_service
from AIRAGAgent.services.access_control_service import access_control_service
from AIRAGAgent.services.auth_service import authenticate_user, create_access_token, register_user
from AIRAGAgent.services.health_service import health_check
from AIRAGAgent.services.conversation_service import conversation_service
from AIRAGAgent.utils.logger_handler import logger

router = APIRouter(prefix="/api")
knowledge_service = KnowledgeBaseService()
rag_service = RagSummarizeService()


def require_admin(user: User) -> None:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")


@router.get("/health")
def health():
    return health_check()


@router.post("/auth/login", response_model=ApiResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid username or password")
    return ApiResponse(
        data=LoginResponse(
            access_token=create_access_token(user),
            username=user.username,
            tenant_id=user.tenant_id,
            role=user.role,
        )
    )


@router.post("/auth/register", response_model=ApiResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, request.username, request.password, request.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ApiResponse(
        data=LoginResponse(
            access_token=create_access_token(user),
            username=user.username,
            tenant_id=user.tenant_id,
            role=user.role,
        )
    )


@router.get("/auth/me", response_model=ApiResponse)
def me(current_user: User = Depends(get_current_user)):
    return ApiResponse(
        data=UserResponse(
            id=current_user.id,
            username=current_user.username,
            tenant_id=current_user.tenant_id,
            role=current_user.role,
        )
    )


@router.get("/knowledge/stats", response_model=ApiResponse)
def knowledge_stats(current_user: User = Depends(get_current_user)):
    require_admin(current_user)
    try:
        return ApiResponse(data=knowledge_service.stats())
    except Exception as exc:
        logger.error("[api] knowledge stats failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/knowledge/ingest", response_model=ApiResponse)
def ingest_knowledge(request: KnowledgeIngestRequest, current_user: User = Depends(get_current_user)):
    require_admin(current_user)
    try:
        return ApiResponse(data=knowledge_service.ingest(force=request.force))
    except Exception as exc:
        logger.error("[api] knowledge ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/knowledge/ingest/async", response_model=ApiResponse)
def ingest_knowledge_async(request: KnowledgeIngestRequest, current_user: User = Depends(get_current_user)):
    require_admin(current_user)
    try:
        return ApiResponse(
            data=knowledge_service.start_ingest_task(
                force=request.force,
                user_id=current_user.id,
                tenant_id="global",
            )
        )
    except Exception as exc:
        logger.error("[api] async knowledge ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/knowledge/tasks/{task_id}", response_model=ApiResponse)
def knowledge_task(task_id: str, current_user: User = Depends(get_current_user)):
    require_admin(current_user)
    task = knowledge_service.get_task(task_id, user_id=current_user.id, tenant_id="global")
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return ApiResponse(data=task)


@router.post("/rag/ask", response_model=ApiResponse)
def rag_ask(request: RagRequest, current_user: User = Depends(get_current_user)):
    decision = access_control_service.can_access_query(request.query, current_user.role)
    if not decision.allowed:
        return ApiResponse(data={"answer": decision.message, "sources": []})
    try:
        return ApiResponse(data=rag_service.answer(request.query, top_k=request.top_k))
    except Exception as exc:
        logger.error("[api] rag ask failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat", response_model=ApiResponse)
def chat(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        response = chat_service.answer(
            query=request.query,
            session_id=request.session_id,
            use_agent=request.use_agent,
            history=request.history,
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            user_role=current_user.role,
            db=db,
        )
        return ApiResponse(data=response)
    except Exception as exc:
        logger.error("[api] chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        return StreamingResponse(
            chat_service.stream_answer(
                query=request.query,
                session_id=request.session_id,
                use_agent=request.use_agent,
                history=request.history,
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                user_role=current_user.role,
                db=db,
            ),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        logger.error("[api] chat stream failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/sessions/{session_id}", response_model=ApiResponse)
def clear_session(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation_service.clear(db, current_user.id, session_id)
    chat_service.clear_context(session_id)
    return ApiResponse(message="session cleared")


@router.get("/conversations", response_model=ApiResponse)
def conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = conversation_service.list_conversations(db, current_user.id)
    return ApiResponse(
        data=[
            ConversationResponse(
                session_id=row.session_id,
                title=row.title,
                created_at=row.created_at.isoformat(),
                updated_at=row.updated_at.isoformat(),
            )
            for row in rows
        ]
    )


@router.get("/conversations/{session_id}/messages", response_model=ApiResponse)
def conversation_messages(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = conversation_service.messages(db, current_user.id, session_id)
    return ApiResponse(
        data=[
            MessageResponse(
                role=row.role,
                content=row.content,
                created_at=row.created_at.isoformat(),
            )
            for row in rows
        ]
    )
