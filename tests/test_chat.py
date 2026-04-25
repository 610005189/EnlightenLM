import requests

url = "http://localhost:8000/inference"
headers = {"Content-Type": "application/json"}

test_messages = [
    "请用Markdown格式回复，包含标题、列表和代码块",
    "解释一下Python的列表推导式",
    "用代码示例说明",
]

for msg in test_messages:
    print(f"\n>>> {msg}")
    response = requests.post(url, headers=headers, json={"text": msg})
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['output'][:200]}...")
    else:
        print(f"Error: {response.status_code}")
    import time
    time.sleep(1)