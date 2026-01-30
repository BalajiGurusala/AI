import requests
import json

def test():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "borrower_id": 1071795,
        "loan_amnt": 5600.0,
        "term": "60 months",
        "purpose": "small_business"
    }
    
    print(f"Testing API with High Risk Borrower 1071795 (Grade F)...")
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test()
