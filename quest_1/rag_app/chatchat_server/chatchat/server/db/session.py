from functools import wraps
from contextlib import contextmanager
from chatchat.server.db.base import SessionLocal
from sqlalchemy.orm import Session


@contextmanager
def session_scope() -> Session:
    """Context manager for automatically obtaining Session, avoiding errors"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper


def get_db() -> SessionLocal:
    """Function to get a database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> SessionLocal:
    """Function to get a database session"""
    db = SessionLocal()
    return db
