# ANLI Multi-Class Classification with DeBERTa

End-to-end Natural Language Inference (NLI) classification system comparing two training approaches on the Adversarial NLI (ANLI) Round 2 dataset.

## Project Overview

**Task**: 3-way classification (entailment, neutral, contradiction)  
**Dataset**: ANLI Round 2 (45,460 train / 1,000 val / 1,000 test)  
**Models**: 
- Base DeBERTa-v3 (trained from scratch): 48% accuracy
- Transfer Learning DeBERTa-v3-mnli-fever-anli: 52% accuracy

### What is Natural Language Inference?

NLI determines the logical relationship between two sentences:
- **Entailment**: Hypothesis is true given the premise
- **Neutral**: Cannot determine truth from premise alone  
- **Contradiction**: Hypothesis is false given the premise

---

**Note on Models:** The trained model directories (`anli_model/` and `anli_model_TF/`) are excluded from Git due to file size constraints (700MB each). These directories must be present locally to run the API.

---

### Prerequisites

- Python 3.11+
- Docker
- Trained models in `anli_model/` and `anli_model_TF/` directories

### Docker Deployment
```bash
# Clone repository
git clone git@github.com:im-rohan/ANLI_Classification_Round_2.git
cd anli-classification

# Ensure model directories exist with trained models:
# - anli_model/
# - anli_model_TF/

# Build and run
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Local Development
```bash
# Clone repository
git clone git@github.com:im-rohan/ANLI_Classification_Round_2.git
cd anli-classification

# Create virtual environment (Pyenv used here)
pyenv virtualenv 3.11.9 <env_name>
pyenv local <env_name>

# Install dependencies
pip install -r requirements.txt

# Run API
python app.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🔍 Model Comparison API

The deployed API allows side-by-side comparison of both models:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed system status |
| `/predict` | POST | Make predictions (single or dual model) |
| `/model-info` | GET | Model details and comparison |
| `/docs` | GET | Interactive API documentation |

### Example Usage

**Compare Both Models:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "A woman is walking her dog in the park.",
    "hypothesis": "A person is outdoors with an animal.",
    "model_type": "both"
  }'
```

**Response:**
```json
{
  "premise": "A woman is walking her dog in the park.",
  "hypothesis": "A person is outdoors with an animal.",
  "base_model": {
    "prediction": "entailment",
    "confidence": 0.82,
  },
  "transfer_model": {
    "prediction": "entailment",
    "confidence": 0.94,
  },
  "agreement": true
}
```

**Test Single Model:**
```bash
# Base model only
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "The meeting is at 3 PM",
    "hypothesis": "The meeting is in the afternoon",
    "model_type": "base"
  }'

# Transfer model only
# Change "model_type": "transfer"
```

---

### Docker Commands
```bash
# Build image
docker build -t anli-api .

# Run container
docker run -p 8000:8000 anli-api

# Use docker-compose (for quick start)
docker-compose up --build

# Stop container
docker-compose down

# View logs
docker logs <container-id>
```

---

## 🧪 Testing
```bash
# Run test suite
python test_api.py

# Manual testing
curl http://localhost:8000/health

# Interactive testing
# Visit: http://localhost:8000/docs
```