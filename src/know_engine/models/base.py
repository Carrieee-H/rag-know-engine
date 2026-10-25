from datetime import datetime

from sqlalchemy import BigInteger, DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

# Deterministic constraint names so Alembic migrations stay stable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def bigint_pk() -> Mapped[int]:
    return mapped_column(BigInteger, primary_key=True, autoincrement=True)


class AuditMixin:
    """Columns shared by every table, mirroring the Java version.

    lock_version is mapped as SQLAlchemy's version counter (optimistic lock);
    deleted is the soft-delete flag.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    lock_version: Mapped[int] = mapped_column(nullable=False, server_default="0")
    deleted: Mapped[bool] = mapped_column(nullable=False, server_default="false")

    @declared_attr.directive
    def __mapper_args__(cls) -> dict:
        # eager_defaults: fetch server-generated values (updated_at) with RETURNING on every
        # write, so objects stay fully loaded after their session closes.
        return {"version_id_col": cls.__table__.c.lock_version, "eager_defaults": True}
