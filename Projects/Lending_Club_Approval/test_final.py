import requests
import json

def test():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "borrower_id": 1069057,
        "loan_amnt": 10000.0,
        "term": "36 months",
        "purpose": "other"
    }
    
    print(f"Testing API with Charged Off borrower 1069057...")
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(json.dumps(resp.json(), indent=2))
        else:
            print(resp.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test()
