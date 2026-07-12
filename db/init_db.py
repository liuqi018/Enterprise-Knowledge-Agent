from sqlalchemy.orm import Session

from AIRAGAgent.config.settings import settings
from AIRAGAgent.db.models import Tenant, User
from AIRAGAgent.db.session import Base, SessionLocal, engine
from AIRAGAgent.services.auth_service import hash_password


def init_mysql() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_admin(db)
    finally:
        db.close()


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
