import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from api.router import router
from AIRAGAgent.db.init_db import init_mysql

FRONTEND_DIR = os.path.join(CURRENT_DIR, "frontend")

app = FastAPI(
    title="Enterprise AI Agent Platform",
    version="1.0.0",
    description="RAG + Agent enterprise knowledge intelligent platform",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def startup():
    init_mysql()


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# uvicorn app:app --reload
