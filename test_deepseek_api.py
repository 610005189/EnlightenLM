"""
DeepSeek API 测试脚本
测试 DeepSeek API 客户端和 EnlightenLM 集成
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.api.deepseek_client import DeepSeekAPIClient, create_deepseek_client


def test_deepseek_api():
    """测试 DeepSeek API"""
    print("=== 测试 DeepSeek API ===\n")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 错误: DEEPSEEK_API_KEY 环境变量未设置")
        print("请运行以下命令设置环境变量:")
        print('  Windows: $env:DEEPSEEK_API_KEY="your-api-key"')
        print('  Linux/Mac: export DEEPSEEK_API_KEY="your-api-key"')
        return False

    print(f"✅ API Key 已设置: {api_key[:8]}...{api_key[-4:]}")

    try:
        client = DeepSeekAPIClient()
        print("✅ DeepSeekAPIClient 创建成功")
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        return False

    print("\n检查 API 可用性...")
    is_available = client.is_available()
    print(f"API 可用: {is_available}")

    if not is_available:
        print("⚠️  API 不可用，可能是网络问题或 API key 无效")

    print("\n测试生成...")
    test_prompts = [
        "你好，请介绍一下自己",
        "解释一下什么是机器学习",
        "写一个关于人工智能的短故事"
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")

        response, latency = client.generate(prompt, max_tokens=200)

        print(f"输出: {response[:200]}...")
        print(f"延迟: {latency:.2f}秒")

    print("\n测试流式生成...")
    print("输入: 讲一个笑话")

    print("输出: ", end="", flush=True)
    for chunk in client.generate_stream("讲一个笑话", max_tokens=100):
        print(chunk, end="", flush=True)
    print()

    print("\n获取模型信息...")
    model_info = client.get_model_info()
    if "error" in model_info:
        print(f"❌ 获取模型信息失败: {model_info['error']}")
    else:
        print(f"✅ 获取模型信息成功")
        if "data" in model_info:
            print(f"可用模型数量: {len(model_info['data'])}")

    print("\n=== 测试完成 ===")
    return True


if __name__ == "__main__":
    success = test_deepseek_api()
    sys.exit(0 if success else 1)
