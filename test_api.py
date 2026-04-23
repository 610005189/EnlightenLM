import requests

url = "http://localhost:8000/inference"
headers = {"Content-Type": "application/json"}
data = {"text": "你好，请介绍一下自己"}

print("Testing API call...")
response = requests.post(url, headers=headers, json=data)
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")