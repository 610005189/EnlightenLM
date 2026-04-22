import requests

# 测试健康检查端点
print("Testing health check endpoint...")
response = requests.get("http://localhost:8000/health")
print(f"Health check response: {response.status_code} - {response.json()}")

# 测试推理端点
print("\nTesting inference endpoint...")
test_data = {
    "text": "你好，介绍一下你自己"
}
response = requests.post("http://localhost:8000/inference", json=test_data)
print(f"Inference response: {response.status_code}")
if response.status_code == 200:
    print(f"Response content: {response.json()}")
else:
    print(f"Error: {response.text}")

# 测试哈希链验证端点
print("\nTesting hash chain verification endpoint...")
response = requests.get("http://localhost:8000/audit/verify")
print(f"Verification response: {response.status_code} - {response.json()}")

# 测试威胁检测统计端点
print("\nTesting threat detection statistics endpoint...")
response = requests.get("http://localhost:8000/audit/threat/statistics")
print(f"Statistics response: {response.status_code} - {response.json()}")
