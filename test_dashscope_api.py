"""
DashScope API 测试脚本
测试阿里云百炼 API 客户端
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enlighten.api.dashscope_client import DashScopeAPIClient, create_dashscope_client


def test_dashscope_api():
    """测试阿里云百炼 API"""
    print("=== 测试阿里云百炼 API ===\n")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: DASHSCOPE_API_KEY 环境变量未设置")
        print("请运行以下命令设置环境变量:")
        print('  Windows: $env:DASHSCOPE_API_KEY="your-api-key"')
        print('  Linux/Mac: export DASHSCOPE_API_KEY="your-api-key"')
        return False

    print(f"✅ API Key 已设置: {api_key[:8]}...{api_key[-4:]}")

    try:
        client = DashScopeAPIClient()
        print("✅ DashScopeAPIClient 创建成功")
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        return False

    print("\n检查 API 可用性...")
    is_available = client.is_available()
    print(f"API 可用: {is_available}")

    print("\n获取模型列表...")
    model_list = client.get_model_list()
    if "error" in model_list:
        print(f"❌ 获取模型列表失败: {model_list['error']}")
    else:
        print(f"✅ 获取模型列表成功")
        if "data" in model_list:
            print(f"可用模型数量: {len(model_list['data'])}")

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

        print(f"输出: {response[:300]}..." if len(response) > 300 else f"输出: {response}")
        print(f"延迟: {latency:.2f}秒")

    print("\n测试带系统提示的生成...")
    system_prompt = "你是一个 helpful 的 AI 助手"
    user_prompt = "解释量子计算的基本原理"

    response, latency = client.generate_with_system(
        system_prompt, user_prompt, max_tokens=300
    )

    print(f"系统提示: {system_prompt}")
    print(f"用户提示: {user_prompt}")
    print(f"输出: {response[:300]}..." if len(response) > 300 else f"输出: {response}")
    print(f"延迟: {latency:.2f}秒")

    print("\n=== 测试完成 ===")
    return True


if __name__ == "__main__":
    success = test_dashscope_api()
    sys.exit(0 if success else 1)
