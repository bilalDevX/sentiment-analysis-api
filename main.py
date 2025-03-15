from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, Integer, String, JSON
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Database Configuration (Async SQLAlchemy with SQLite)
DATABASE_URL = "sqlite+aiosqlite:///./sentiment.db"
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Define Base for SQLAlchemy Models
class Base(DeclarativeBase):
    pass

# Load Emotion Analysis Model
sentiment_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Database Model
class Sentiment(Base):
    __tablename__ = "sentiments"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    emotions = Column(JSON, nullable=False)  # Store multiple emotions & scores as JSON

# Pydantic Schemas
class SentimentCreate(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    id: int
    text: str
    emotions: dict  # Dictionary storing emotions & confidence scores

    class Config:
        from_attributes = True

# Dependency for Database Session
async def get_db():
    async with SessionLocal() as session:
        yield session

# Create Database Tables
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Startup Event
@app.on_event("startup")
async def on_startup():
    await init_db()

# API Endpoints
@app.post("/sentiment/", response_model=SentimentResponse)
async def analyze_sentiment(input: SentimentCreate, db: AsyncSession = Depends(get_db)):
    predictions = sentiment_model(input.text)  # Get multiple emotion predictions
    
    # Convert list of dictionaries into a structured dictionary
    emotions_dict = {emotion["label"]: round(emotion["score"], 3) for emotion in predictions[0]}

    db_sentiment = Sentiment(text=input.text, emotions=emotions_dict)
    db.add(db_sentiment)
    await db.commit()
    await db.refresh(db_sentiment)

    return db_sentiment

@app.get("/sentiment/{id}", response_model=SentimentResponse)
async def get_sentiment(id: int, db: AsyncSession = Depends(get_db)):
    result = await db.get(Sentiment, id)

    if not result:
        raise HTTPException(status_code=404, detail="Sentiment not found")

    return result
