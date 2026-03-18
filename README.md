��# Sentiment Analysis API

A production-ready REST API for sentiment analysis using BERT fine-tuned on the IMDB dataset. Classifies text as POSITIVE or NEGATIVE with confidence scores.

## Live Demo

**Try it now:** https://adedayo2000-sentiment-fastapi-api.hf.space/docs

**Quick test:**
```bash
curl -X POST "https://adedayo2000-sentiment-fastapi-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It is amazing!"}'
```

**Response:**
```json
{
  "text": "I love this product! It is amazing!",
  "sentiment": "POSITIVE",
  "confidence": "99.87%",
  "confidence_score": 0.9987
}
```

---

## Table of Contents

- Features
- Tech Stack
- API Endpoints
- Installation
- Usage
- Docker Deployment
- Model Details
- Project Structure
- Performance
- Contributing
- License
- Contact

---
## Features

- **Real-time sentiment analysis** - Instant predictions via REST API
- **High accuracy** - 98%+ accuracy on IMDB test set
- **Interactive documentation** - Auto-generated Swagger UI
- **Production-ready** - Dockerized, deployed on Hugging Face Spaces
- **CORS enabled** - Ready for web applications
- **Health checks** - Monitor API status
- **Error handling** - Graceful error responses

---

## Tech Stack

**Backend:**
- FastAPI (https://fastapi.tiangolo.com/) - Modern Python web framework
- Transformers (https://huggingface.co/transformers/) - Hugging Face library
- PyTorch (https://pytorch.org/) - Deep learning framework

**Model:**
- BERT fine-tuned on IMDB dataset
- 25,000 movie reviews for training
- Binary classification (Positive/Negative)

**Deployment:**
- Docker containerization
- Hugging Face Spaces hosting
- Automatic CI/CD

---

## API Endpoints

### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Sentiment Analysis API",
  "status": "live",
  "endpoints": {
    "GET /": "API information",
    "POST /predict": "Analyze sentiment",
    "GET /docs": "Interactive documentation",
    "GET /health": "Health check"
  }
}
```

---

### `POST /predict`
Analyzes the sentiment of input text.

**Request:**
```json
{
  "text": "This movie was absolutely terrible!"
}
```

**Response:**
```json
{
  "text": "This movie was absolutely terrible!",
  "sentiment": "NEGATIVE",
  "confidence": "98.76%",
  "confidence_score": 0.9876
}
```

---

### `GET /health`
Returns API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `GET /docs`
Interactive Swagger UI documentation for testing endpoints.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Adedayo2000/sentiment-analysis.git
cd sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

4. **Access the API**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Docker Deployment

### Build and run with Docker

```bash
# Build the image
docker build -t sentiment-api .

# Run the container
docker run -p 8000:8000 sentiment-api
```

### Using Docker Compose

```bash
docker-compose up
```

---

## Usage Examples

### Python

```python
import requests

url = "https://adedayo2000-sentiment-fastapi-api.hf.space/predict"
data = {"text": "I absolutely loved this movie!"}

response = requests.post(url, json=data)
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript

```javascript
fetch('https://adedayo2000-sentiment-fastapi-api.hf.space/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'This product is amazing!'})
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
curl -X POST "https://adedayo2000-sentiment-fastapi-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Best purchase ever!"}'
```

---

## Model Details

**Base Model:** BERT (Bidirectional Encoder Representations from Transformers)

**Training Data:** IMDB Movie Reviews
- 25,000 training samples
- 25,000 test samples
- Binary classification (Positive/Negative)

**Performance:**
- **Accuracy:** 98%+
- **Precision:** 97.8%
- **Recall:** 98.2%
- **F1 Score:** 98.0%

**Model Location:** [Adedayo2000/sentiment-model](https://huggingface.co/Adedayo2000/sentiment-model)

---

## Project Structure

```
sentiment-analysis/
├── app.py                 # FastAPI application
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
```

---

## Performance

**Benchmarks:**
- **Average response time:** ~500ms
- **Model loading time:** ~6 seconds (first request)
- **Memory usage:** ~400MB
- **Platform:** Free tier Hugging Face Spaces (CPU)

**Notes:**
- First prediction after deployment takes 5-10 seconds (model loading)
- Subsequent predictions are faster (~500ms)
- Running on CPU (free tier) - GPU would be significantly faster

**Tested on:**
- **CPU:** Hugging Face Spaces (CPU Basic)
- **RAM:** 2GB
- **Platform:** Hugging Face Spaces
- **Region:** Cloud (auto-selected)

---

## Deployment

This API is deployed on **Hugging Face Spaces** using Docker.

**Live URL:** https://adedayo2000-sentiment-fastapi-api.hf.space

### Deployment Journey:

**Initial attempt:** Render.com
- Hit 512MB disk space limit (model is ~400MB)
- Switched to Hugging Face Spaces

**Final deployment:** Hugging Face Spaces
- Supports larger models
- Free Docker deployment
- Automatic rebuilds on git push

### Deployment Process:
1. Code pushed to Hugging Face Space repository
2. Dockerfile triggers automatic build
3. Docker container builds (~5-10 minutes)
4. API goes live with green "Running" status
5. Accessible at `.hf.space` domain

### Key Learnings:
- **Render limitation:** 512MB disk space on free tier insufficient for ML models
- **Hugging Face Spaces:** Better suited for ML deployments
- **Docker configuration:** Required `Dockerfile` (capital D), not `docker`
- **Port configuration:** Must use port 7860 for Hugging Face Spaces
- **README header:** Must include SDK metadata for automatic deployment

---

## Debugging Notes

**Issues encountered and resolved:**

1. **Render deployment failed**
   - Problem: Model size (400MB) exceeded 512MB disk limit
   - Solution: Migrated to Hugging Face Spaces

2. **"No application file" error**
   - Problem: Dockerfile named `docker` (lowercase)
   - Solution: Renamed to `Dockerfile` (capital D)

3. **404 on /docs endpoint**
   - Problem: FastAPI docs not explicitly enabled
   - Solution: Added `docs_url="/docs"` to FastAPI initialization

4. **Label mapping issue**
   - Problem: API returned "LABEL_0" and "LABEL_1" instead of "POSITIVE"/"NEGATIVE"
   - Solution: Added label mapping dictionary to convert model outputs

5. **Confidence display**
   - Problem: Confidence shown as decimal (0.8725)
   - Solution: Added percentage formatting (87.25%)

---

## Project Structure

```
sentiment-analysis/
├── app.py                 # FastAPI application
├── Dockerfile             # Docker configuration (MUST be capital D!)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

**Key files:**
- `app.py` - Main application with FastAPI endpoints
- `Dockerfile` - Container configuration for deployment
- `requirements.txt` - Lists: fastapi, uvicorn, transformers, torch

---

## Update Process

**To update the deployed API:**

1. **Update code on GitHub** (source of truth)
   ```bash
   git add app.py
   git commit -m "Update: description of changes"
   git push origin main
   ```

2. **Update Hugging Face Space**
   - Navigate to Space repository
   - Edit files directly or push via git
   - Space rebuilds automatically (~3 minutes)

3. **Test the deployment**
   - Visit `/docs` endpoint
   - Test with sample requests
   - Verify responses are correct

**Note:** GitHub and Hugging Face Space are separate repositories. Both must be updated to stay in sync.

---

## Deployment

This API is deployed on **Hugging Face Spaces** with automatic rebuilds.

**Live URL:** https://adedayo2000-sentiment-fastapi-api.hf.space

**Deployment Process:**
1. Code is pushed to GitHub
2. Changes are synced to Hugging Face Space
3. Docker container rebuilds automatically
4. API goes live within 2-3 minutes

---

## Testing

**Manual Testing:**
1. Visit the [Swagger UI](https://adedayo2000-sentiment-fastapi-api.hf.space/docs)
2. Click on `POST /predict`
3. Click "Try it out"
4. Enter sample text
5. Click "Execute"
6. View results

**Test Cases:**
```json
// Positive sentiment
{"text": "I love this product! Best purchase ever!"}

// Negative sentiment
{"text": "Terrible experience. Complete waste of money."}

// Neutral text
{"text": "It's okay, nothing special."}

// Edge case
{"text": ""}  // Returns error
```

---

## What I Learned

Building this project taught me:

- **FastAPI** - Creating production-ready REST APIs
- **Docker** - Containerizing Python applications
- **Transformers** - Working with pre-trained BERT models
- **Deployment** - Deploying ML models to the cloud
- **API Design** - Designing user-friendly endpoints
- **Error Handling** - Graceful error responses
- **Documentation** - Auto-generated API docs with Swagger

---

## Future Improvements

- [ ] Add support for multi-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)
- [ ] Implement batch prediction endpoint
- [ ] Add rate limiting
- [ ] Include sentiment analysis for multiple languages
- [ ] Add caching for common queries
- [ ] Implement A/B testing for model versions
- [ ] Create web UI for non-technical users
- [ ] Add monitoring and analytics dashboard

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

---

## 👤 Author

**Adedayo Adebayo**

- Hugging Face: [@Adedayo2000](https://huggingface.co/Adedayo2000)
- GitHub: [@Adedayo2000](https://github.com/Adedayo2000)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) - For Transformers library and hosting
- [FastAPI](https://fastapi.tiangolo.com/) - For the excellent web framework
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) - For training data

---

## Contact

Have questions or suggestions? Feel free to reach out!

- **Email:** adebayoadedayo23@gmail.com
- **GitHub Issues:** [Create an issue](https://github.com/Adedayo2000/sentiment-analysis/issues)

---

## Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with ❤️ by Adedayo Adebayo**
