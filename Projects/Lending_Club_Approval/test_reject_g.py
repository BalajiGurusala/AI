import requests
import json

def test():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "borrower_id": 1058624,
        "loan_amnt": 12375.0,
        "term": "60 months",
        "purpose": "other"
    }
    
    print(f"Testing API with Ultra-High Risk Borrower 1058624 (Grade G)...")
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test()
