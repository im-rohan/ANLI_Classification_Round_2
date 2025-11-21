import requests
import json

BASE_URL = "http://localhost:8000"

def test_both_models():
    data = {
        "premise": "A dog is running in the park.",
        "hypothesis": "An animal is outside.",
        "model_type": "both"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    result = response.json()
    
    print("Model Comparison:")
    print(f"\nPremise: {result['premise']}")
    print(f"Hypothesis: {result['hypothesis']}")
    
    print(f"\n")
    print("BASE MODEL:")
    print(f"  Prediction: {result['base_model']['prediction']}")
    print(f"  Confidence: {result['base_model']['confidence']:.2%}")
    
    print(f"\n")
    print("TRANSFER MODEL:")
    print(f"  Prediction: {result['transfer_model']['prediction']}")
    print(f"  Confidence: {result['transfer_model']['confidence']:.2%}")
    
    print(f"\n")
    print(f"Models Agree: {'Yes' if result['agreement'] else 'No'}")
    print(f"\n")

if __name__ == "__main__":
    test_both_models()