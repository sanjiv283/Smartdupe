from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    files = relationship("File", back_populates="owner")


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    file_type = Column(String)
    path = Column(String, nullable=False)

    partial_hash = Column(String)
    full_hash = Column(String)

    is_duplicate = Column(Boolean, default=False)
    duplicate_of_id = Column(Integer, ForeignKey("files.id"), nullable=True)

    # AI similarity
    similarity_score = Column(Float, nullable=True)        # 0.0 – 1.0 cosine score vs nearest file
    similar_to_filename = Column(String, nullable=True)    # name of the most similar file

    # Clustering
    cluster_id = Column(Integer, nullable=True)
    embedding_vector = Column(String, nullable=True)       # JSON-serialised list[float]

    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="files")
