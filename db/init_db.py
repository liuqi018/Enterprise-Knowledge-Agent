from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from AIRAGAgent.config.settings import settings
from AIRAGAgent.db.models import Tenant, User
from AIRAGAgent.db.session import Base, SessionLocal, engine
from AIRAGAgent.services.auth_service import hash_password
from AIRAGAgent.utils.logger_handler import logger


def init_mysql() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema_upgrades()
    db = SessionLocal()
    try:
        seed_admin(db)
    finally:
        db.close()


def ensure_schema_upgrades() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "conversations" not in table_names:
        return

    conversation_columns = {column["name"] for column in inspector.get_columns("conversations")}
    if "summary" not in conversation_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT NULL"))
        logger.info("[db] added conversations.summary column")


def seed_admin(db: Session) -> None:
    tenant = db.query(Tenant).filter(Tenant.tenant_id == settings.ADMIN_TENANT_ID).first()
    if not tenant:
        db.add(Tenant(tenant_id=settings.ADMIN_TENANT_ID, name=settings.ADMIN_TENANT_ID))

    user = db.query(User).filter(User.username == settings.ADMIN_USERNAME).first()
    if not user:
        db.add(
            User(
                username=settings.ADMIN_USERNAME,
                password_hash=hash_password(settings.ADMIN_PASSWORD),
                tenant_id=settings.ADMIN_TENANT_ID,
                role="admin",
                is_active=True,
            )
        )
    db.commit()
