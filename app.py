from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment of text using BERT fine-tuned on IMDB dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading sentiment analysis model...")
try:
    classifier = pipeline("sentiment-analysis", model="Adedayo2000/sentiment-model")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
}

class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!"
            }
        }

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API",
        "status": "live",
        "model": "Adedayo2000/sentiment-model",
        "endpoints": {
            "GET /": "API information",
            "POST /predict": "Analyze sentiment of text",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

@app.post("/predict")
def predict(input: TextInput):
    if not input.text or not input.text.strip():
        return {
            "error": "Text cannot be empty",
            "example": {"text": "I love this product!"}
        }
    
    if classifier is None:
        return {
            "error": "Model not loaded",
            "status": "unavailable"
        }
    
    try:
        result = classifier(input.text)[0]
        
        raw_label = result["label"]
        sentiment = LABEL_MAP.get(raw_label, raw_label)
        
        confidence_score = result["score"]
        confidence_percentage = f"{confidence_score * 100:.2f}%"
        
        return {
            "text": input.text,
            "sentiment": sentiment,
            "confidence": confidence_percentage,
            "confidence_score": round(confidence_score, 4)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "error": "Prediction failed",
            "details": str(e)
        }
        
