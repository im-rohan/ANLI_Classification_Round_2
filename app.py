from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ANLI Classification API - Dual Model Comparison",
    description="Compare Base DeBERTa vs Transfer Learning (NLI-pretrained) on ANLI R2",
    version="1.0.0"
)

# Model paths
BASE_MODEL_PATH = "./anli_model"           # Base DeBERTa model
TRANSFER_MODEL_PATH = "./anli_model_TF"    # Transfer learning model
MAX_LENGTH = 256
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# Load both models
logger.info("Loading models")

if torch.backends.mps.is_available(): # check for macOS GPU
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check for NVIDIA GPU or fallback to CPU

try:
    # Load Base Model
    logger.info(f"Loading base model from {BASE_MODEL_PATH}")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model_base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH)
    model_base.eval()
    model_base.to(device)
    logger.info(f"Base model loaded on {device}")
    
    # Load Transfer Learning Model
    logger.info(f"Loading transfer learning model from {TRANSFER_MODEL_PATH}")
    tokenizer_transfer = AutoTokenizer.from_pretrained(TRANSFER_MODEL_PATH)
    model_transfer = AutoModelForSequenceClassification.from_pretrained(TRANSFER_MODEL_PATH)
    model_transfer.eval()
    model_transfer.to(device)
    logger.info(f"Transfer learning model loaded on {device}")
    
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Request/Response models
class NLIRequest(BaseModel):
    premise: str
    hypothesis: str
    model_type: Literal["base", "transfer", "both"] = "both"
    
    class Config:
        schema_extra = {
            "example": {
                "premise": "A woman is walking her dog in the park.",
                "hypothesis": "A person is outdoors with an animal.",
                "model_type": "both"
            }
        }

class SingleModelResponse(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class DualModelResponse(BaseModel):
    premise: str
    hypothesis: str
    base_model: SingleModelResponse
    transfer_model: SingleModelResponse
    agreement: bool

# Helper function for prediction
def predict_single(premise: str, hypothesis: str, model, tokenizer, model_name: str) -> SingleModelResponse:
    """Make prediction with a single model"""
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return SingleModelResponse(
        model_name=model_name,
        prediction=label_map[predicted_class],
        confidence=float(probabilities[0][predicted_class].item()),
        probabilities={
            label_map[i]: float(probabilities[0][i].item())
            for i in range(3)
        }
    )

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "base": "DeBERTa-v3-base (trained from scratch)",
            "transfer": "DeBERTa-v3-base-mnli-fever-anli (transfer learning)"
        },
        "task": "Natural Language Inference - ANLI Round 2",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "base_model_loaded": model_base is not None,
        "transfer_model_loaded": model_transfer is not None,
        "device": str(device),
        "gpu_available": torch.cuda.is_available() or torch.backends.mps.is_available()
    }

@app.post("/predict")
async def predict(request: NLIRequest):
    """
    Predict NLI relationship using one or both models.
    
    Args:
        request: NLIRequest with premise, hypothesis, and model_type
        
    Returns:
        Predictions from selected model(s)
    """
    try:
        if request.model_type == "base":
            result = predict_single(
                request.premise, 
                request.hypothesis,
                model_base,
                tokenizer_base,
                "Base DeBERTa"
            )
            return result
            
        elif request.model_type == "transfer":
            result = predict_single(
                request.premise,
                request.hypothesis,
                model_transfer,
                tokenizer_transfer,
                "Transfer Learning"
            )
            return result
            
        else:  # both
            base_result = predict_single(
                request.premise,
                request.hypothesis,
                model_base,
                tokenizer_base,
                "Base DeBERTa"
            )
            
            transfer_result = predict_single(
                request.premise,
                request.hypothesis,
                model_transfer,
                tokenizer_transfer,
                "Transfer Learning"
            )
            
            return DualModelResponse(
                premise=request.premise,
                hypothesis=request.hypothesis,
                base_model=base_result,
                transfer_model=transfer_result,
                agreement=(base_result.prediction == transfer_result.prediction)
            )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get information about both models"""
    return {
        "base_model": {
            "name": "microsoft/deberta-v3-base",
            "description": "Trained from scratch on ANLI R2",
            "training_epochs": 7,
            "test_accuracy": "48%",
            "approach": "Clean slate learning"
        },
        "transfer_model": {
            "name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "description": "Pre-trained on MNLI, FEVER, ANLI R1, fine-tuned on R2",
            "training_epochs": 12,
            "test_accuracy": "52%",
            "approach": "Transfer learning with aggressive fine-tuning"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)