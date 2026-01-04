"""Database configuration and session management using SQLAlchemy"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from ..config.settings import get_settings

# Get settings
settings = get_settings()

# Configure engine based on database type
# SQLite uses StaticPool (no connection pooling), PostgreSQL uses QueuePool
if settings.database_url.startswith("sqlite"):
    # SQLite configuration: StaticPool for single-threaded access
    # check_same_thread=False allows use across threads with proper session management
    engine = create_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    # PostgreSQL/MySQL/other databases: Use connection pooling
    engine = create_engine(
        settings.database_url,
        echo=settings.debug,  # Log SQL queries in debug mode
        # Connection pool settings (for PostgreSQL, MySQL, etc.)
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,         # Number of connections to maintain
        max_overflow=10      # Maximum additional connections
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for declarative models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.

    Yields a database session and ensures it's closed after use.
    This should be used with FastAPI's Depends() for dependency injection.

    Yields:
        Session: SQLAlchemy database session

    Example:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database by creating all tables.

    This function should be called on application startup
    to ensure all tables exist.
    """
    # Import all models here to ensure they're registered with Base
    from .models import synthesis_model, voice_reference_model

    # Create all tables
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data!
    Only use this for testing or resetting the database.
    """
    Base.metadata.drop_all(bind=engine)


def reset_db() -> None:
    """
    Reset database by dropping and recreating all tables.

    WARNING: This will delete all data!
    Only use this for testing or development.
    """
    drop_db()
    init_db()
