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
from AIRAGAgent.utils.trace import elapsed_ms, log_trace, new_trace_id, now_ms, short_text, trace_scope

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
    trace_id = new_trace_id()
    with trace_scope(trace_id):
        start = now_ms()
        log_trace(
            logger,
            "api_request",
            endpoint="/api/rag/ask",
            user=current_user.username,
            role=current_user.role,
            tenant_id=current_user.tenant_id,
            query=short_text(request.query),
        )
        decision = access_control_service.can_access_query(request.query, current_user.role)
        if not decision.allowed:
            log_trace(
                logger,
                "api_response",
                endpoint="/api/rag/ask",
                status="access_denied",
                access_domain=decision.domain,
                access_action=decision.action,
                access_reason=decision.reason,
                elapsed_ms=elapsed_ms(start),
            )
            return ApiResponse(data={"answer": decision.message, "sources": []})
        try:
            response = ApiResponse(data=rag_service.answer(request.query, top_k=request.top_k, trace_id=trace_id))
            log_trace(logger, "api_response", endpoint="/api/rag/ask", status="ok", elapsed_ms=elapsed_ms(start))
            return response
        except Exception as exc:
            logger.error("[api] rag ask failed: %s", exc, exc_info=True)
            log_trace(logger, "api_response", endpoint="/api/rag/ask", status="error", error=f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms(start))
            raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat", response_model=ApiResponse)
def chat(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trace_id = new_trace_id()
    start = now_ms()
    with trace_scope(trace_id):
        try:
            log_trace(
                logger,
                "api_request",
                endpoint="/api/chat",
                user=current_user.username,
                role=current_user.role,
                tenant_id=current_user.tenant_id,
                session_id=request.session_id,
                query=short_text(request.query),
            )
            response = chat_service.answer(
                query=request.query,
                session_id=request.session_id,
                use_agent=request.use_agent,
                history=request.history,
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                user_role=current_user.role,
                db=db,
                trace_id=trace_id,
            )
            log_trace(logger, "api_response", endpoint="/api/chat", status="ok", elapsed_ms=elapsed_ms(start))
            return ApiResponse(data=response)
        except Exception as exc:
            logger.error("[api] chat failed: %s", exc, exc_info=True)
            log_trace(logger, "api_response", endpoint="/api/chat", status="error", error=f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms(start))
            raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trace_id = new_trace_id()
    with trace_scope(trace_id):
        log_trace(
            logger,
            "api_request",
            endpoint="/api/chat/stream",
            user=current_user.username,
            role=current_user.role,
            tenant_id=current_user.tenant_id,
            session_id=request.session_id,
            query=short_text(request.query),
        )
    try:
        response = StreamingResponse(
            chat_service.stream_answer(
                query=request.query,
                session_id=request.session_id,
                use_agent=request.use_agent,
                history=request.history,
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                user_role=current_user.role,
                db=db,
                trace_id=trace_id,
            ),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Trace-Id": trace_id,
            },
        )
        with trace_scope(trace_id):
            log_trace(logger, "api_response", endpoint="/api/chat/stream", status="stream_started")
        return response
    except Exception as exc:
        logger.error("[api] chat stream failed: %s", exc, exc_info=True)
        with trace_scope(trace_id):
            log_trace(logger, "api_response", endpoint="/api/chat/stream", status="error", error=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/messages/{message_id}/retry", response_model=ApiResponse)
def retry_message(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    trace_id = new_trace_id()
    start = now_ms()
    with trace_scope(trace_id):
        try:
            log_trace(
                logger,
                "api_request",
                endpoint="/api/messages/retry",
                user=current_user.username,
                role=current_user.role,
                tenant_id=current_user.tenant_id,
                message_id=message_id,
            )
            response = chat_service.retry_answer(
                assistant_message_id=message_id,
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                user_role=current_user.role,
                db=db,
                trace_id=trace_id,
            )
            log_trace(logger, "api_response", endpoint="/api/messages/retry", status="ok", elapsed_ms=elapsed_ms(start))
            return ApiResponse(data=response)
        except ValueError as exc:
            log_trace(logger, "api_response", endpoint="/api/messages/retry", status="bad_request", error=str(exc), elapsed_ms=elapsed_ms(start))
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("[api] retry message failed: %s", exc, exc_info=True)
            log_trace(logger, "api_response", endpoint="/api/messages/retry", status="error", error=f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms(start))
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
                id=row.id,
                role=row.role,
                content=row.content,
                status=row.status or "success",
                retryable=bool(row.retryable),
                error_message=row.error_message,
                parent_message_id=row.parent_message_id,
                created_at=row.created_at.isoformat(),
            )
            for row in rows
        ]
    )
