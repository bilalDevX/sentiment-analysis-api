from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel
from transformers import pipeline
import asyncio

# FastAPI app
app = FastAPI()

# Database Config (Async SQLAlchemy with SQLite)
DATABASE_URL = "sqlite+aiosqlite:///./sentiment.db"

engine = create_async_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# Sentiment Model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# sentiment_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Database Model
from sqlalchemy import Column, Integer, String, Float

class Sentiment(Base):
    __tablename__ = "sentiment"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

# Pydantic Schemas
class SentimentCreate(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    confidence: float

    class Config:
        from_attributes = True

# Dependency for DB session
async def get_db():
    async with SessionLocal() as session:
        yield session

# Create DB Tables
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
    sentiment = sentiment_model(input.text)[0]

    db_sentiment = Sentiment(text=input.text, sentiment=sentiment['label'], confidence=sentiment['score'])
    db.add(db_sentiment)
    await db.commit()
    await db.refresh(db_sentiment)

    return db_sentiment

@app.get("/sentiment/{id}", response_model=SentimentResponse)
async def get_sentiment(id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Sentiment).where(Sentiment.id == id))
    db_sentiment = result.scalar()

    if not db_sentiment:
        raise HTTPException(status_code=404, detail="Sentiment not found")

    return db_sentiment
