from enlighten.memory.dataset import HallucinationDataset

# 测试数据处理模块
def test_dataset():
    print("测试数据处理模块...")
    
    # 初始化数据集
    dataset = HallucinationDataset("./data")
    
    # 加载数据
    truthful_qa, halu_eval = dataset.load_data()
    print(f"加载了 {len(truthful_qa)} 条 TruthfulQA 数据")
    print(f"加载了 {len(halu_eval)} 条 HaluEval 数据")
    
    # 预处理数据
    truthful_qa_features, truthful_qa_labels = dataset.preprocess(truthful_qa)
    print(f"TruthfulQA 特征形状: {truthful_qa_features.shape}")
    print(f"TruthfulQA 标签形状: {truthful_qa_labels.shape}")
    
    halu_eval_features, halu_eval_labels = dataset.preprocess(halu_eval)
    print(f"HaluEval 特征形状: {halu_eval_features.shape}")
    print(f"HaluEval 标签形状: {halu_eval_labels.shape}")
    
    # 验证特征值
    print("\n特征示例:")
    print(f"TruthfulQA 第一个样本特征: {truthful_qa_features[0]}")
    print(f"TruthfulQA 第一个样本标签: {truthful_qa_labels[0]}")
    print(f"HaluEval 第一个样本特征: {halu_eval_features[0]}")
    print(f"HaluEval 第一个样本标签: {halu_eval_labels[0]}")
    
    print("\n数据处理模块测试通过！")

if __name__ == "__main__":
    test_dataset()