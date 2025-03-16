# Sentiment Analysis API

This FastAPI-based API performs sentiment analysis using a pre-trained emotion classification model and stores the results in an SQLite database using SQLAlchemy with asynchronous support.

## Features

- Analyze sentiment of text input and store results in a database.
- Retrieve sentiment analysis results by ID.
- Uses a pre-trained transformer model for emotion detection.
- Asynchronous database operations with SQLAlchemy and SQLite.

## Requirements

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install fastapi[all] sqlalchemy[asyncio] aiosqlite pydantic transformers torch
```

## Setup and Run

1. Clone this repository and navigate to the project directory.
2. Install dependencies as mentioned above.
3. Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Analyze Sentiment

**POST** `/sentiment/`

#### Request Body:

```json
{
  "text": "I am feeling fantastic today!"
}
```

#### Response:

```json
{
  "id": 5,
  "text": "I am feeling fantastic today!",
  "emotions": {
    "joy": 0.992,
    "surprise": 0.003,
    "fear": 0.002,
    "sadness": 0.001,
    "neutral": 0.001,
    "anger": 0.001,
    "disgust": 0
  }
}
```

### Get Sentiment by ID

**GET** `/sentiment/{id}`

#### Example Response:

```json
{
  "id": 5,
  "text": "I am feeling fantastic today!",
  "emotions": {
    "joy": 0.992,
    "surprise": 0.003,
    "fear": 0.002,
    "sadness": 0.001,
    "neutral": 0.001,
    "anger": 0.001,
    "disgust": 0
  }
}
```

## Project Structure

```
.
├── main.py        # FastAPI application
├── README.md      # API documentation
└── requirements.txt  # List of dependencies
```

## Notes

- The API uses `j-hartmann/emotion-english-distilroberta-base` for emotion classification.
- Sentiments are stored in an SQLite database (`sentiment.db`).
- The application creates the database tables automatically on startup.

## License

This project is open-source and available under the MIT License.

