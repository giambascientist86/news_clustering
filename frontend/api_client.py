import requests

BASE_URL = "http://localhost:8000"

def train_model():
    response = requests.post(f"{BASE_URL}/train/")
    return response.json()

def get_topics():
    response = requests.get(f"{BASE_URL}/topics/")

    print("Status Code:", response.status_code)
    print("Response Text:", response.text)  # Stampa il contenuto
    
    response.raise_for_status() 
    return response.json()
