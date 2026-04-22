import requests

# 测试健康检查端点
print("Testing health check endpoint...")
response = requests.get("http://localhost:8000/health")
print(f"Health check response: {response.status_code} - {response.json()}")

# 测试哈希链验证端点
print("\nTesting hash chain verification endpoint...")
response = requests.get("http://localhost:8000/audit/verify")
print(f"Verification response: {response.status_code} - {response.json()}")

# 测试威胁检测统计端点
print("\nTesting threat detection statistics endpoint...")
response = requests.get("http://localhost:8000/audit/threat/statistics")
print(f"Statistics response: {response.status_code} - {response.json()}")

# 测试重置威胁检测统计端点
print("\nTesting reset threat detection statistics endpoint...")
response = requests.post("http://localhost:8000/audit/threat/reset")
print(f"Reset response: {response.status_code} - {response.json()}")

print("\nBasic API tests completed!")
