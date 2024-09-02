from sqlalchemy import Column, Integer, String, DateTime, JSON, func
from chatchat.server.db.base import Base

class ConversationModel(Base):
    """
    Conversation record model
    """
    __tablename__ = 'conversation'
    id = Column(String(32), primary_key=True, comment='Conversation ID')
    name = Column(String(50), comment='Conversation Name')
    chat_type = Column(String(50), comment='Chat Type')
    create_time = Column(DateTime, default=func.now(), comment='Creation Time')

    def __repr__(self):
        return f"<Conversation(id='{self.id}', name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"
