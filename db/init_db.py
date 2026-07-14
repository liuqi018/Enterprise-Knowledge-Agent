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
    with engine.begin() as connection:
        if "conversations" in table_names:
            conversation_columns = {column["name"] for column in inspector.get_columns("conversations")}
            if "summary" not in conversation_columns:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT NULL"))
                logger.info("[db] added conversations.summary column")

        if "messages" in table_names:
            message_columns = {column["name"] for column in inspector.get_columns("messages")}
            message_upgrades = {
                "status": "ALTER TABLE messages ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT 'success'",
                "error_message": "ALTER TABLE messages ADD COLUMN error_message TEXT NULL",
                "retryable": "ALTER TABLE messages ADD COLUMN retryable BOOL NOT NULL DEFAULT 0",
                "parent_message_id": "ALTER TABLE messages ADD COLUMN parent_message_id INT NULL",
            }
            for column_name, ddl in message_upgrades.items():
                if column_name not in message_columns:
                    connection.execute(text(ddl))
                    logger.info("[db] added messages.%s column", column_name)


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
