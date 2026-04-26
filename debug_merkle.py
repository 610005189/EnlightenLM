"""
调试默克尔树问题
"""

from enlighten.audit import HashChainFactory, MerkleTree

# 创建测试数据
hash_chain = HashChainFactory.create_memory_chain()
for i in range(4):
    hash_chain.append(
        event_type=f"test_event_{i}",
        session_id="test_session",
        data={"value": i, "message": f"Test message {i}"}
    )

# 获取条目
entries = []
for i in range(4):
    entry = hash_chain.get_entry(i)
    if entry:
        entries.append(entry)

# 构建默克尔树
tree = MerkleTree()
root_hash = tree.build_from_entries(entries)
print(f"Root hash: {root_hash}")
print(f"Leaf count: {tree.get_leaf_count()}")

# 打印叶子节点
print("\nLeaf nodes:")
for i, node in enumerate(tree._leaf_nodes):
    print(f"Leaf {i}: hash={node.hash}, entry_id={node.entry_id}")

# 测试生成证明
print("\nTesting proof generation:")
for i in range(4):
    proof = tree.generate_proof(i)
    if proof:
        print(f"Proof for leaf {i}:")
        print(f"  Leaf hash: {proof.leaf_hash}")
        print(f"  Root hash: {proof.root_hash}")
        print(f"  Path: {proof.path}")
        
        # 验证证明
        is_valid = tree.verify_proof(proof)
        print(f"  Verification: {is_valid}")
        
        # 手动验证
        if is_valid:
            print("  Manual verification:")
            current_hash = proof.leaf_hash
            print(f"    Start: {current_hash}")
            for sibling_hash, is_right in proof.path:
                if is_right:
                    new_hash = tree._hash_pair(sibling_hash, current_hash)
                    print(f"    hash({sibling_hash} + {current_hash}) = {new_hash}")
                else:
                    new_hash = tree._hash_pair(current_hash, sibling_hash)
                    print(f"    hash({current_hash} + {sibling_hash}) = {new_hash}")
                current_hash = new_hash
            print(f"    Final: {current_hash}")
            print(f"    Expected: {proof.root_hash}")
            print(f"    Match: {current_hash == proof.root_hash}")

print("\nTree structure:")
def print_tree(node, level=0):
    if not node:
        return
    indent = "  " * level
    print(f"{indent}{node.hash} {'(leaf)' if node.is_leaf() else ''}")
    if node.left:
        print_tree(node.left, level + 1)
    if node.right:
        print_tree(node.right, level + 1)

print_tree(tree._root)
