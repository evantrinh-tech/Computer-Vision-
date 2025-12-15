import requests
import json

print("Testing /health endpoint...")
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

print("Loading model...")
response = requests.post("http://localhost:8000/model/reload?model_path=models/rbfnn_demo_model.pkl")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

print("Testing /predict endpoint...")
data = {
    "data": [
        {
            "timestamp": "2024-01-01T12:00:00",
            "detector_id": "detector_001",
            "volume": 1000,
            "speed": 60,
            "occupancy": 0.3
        },
        {
            "timestamp": "2024-01-01T12:01:00",
            "detector_id": "detector_002",
            "volume": 300,
            "speed": 20,
            "occupancy": 0.8
        }
    ]
}

response = requests.post("http://localhost:8000/predict", json=data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

print("Testing /metrics endpoint...")
response = requests.get("http://localhost:8000/metrics")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")