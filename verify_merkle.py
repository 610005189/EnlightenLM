"""
手动验证默克尔树计算
"""

import hashlib

def hash_pair(left, right):
    combined = f"{left}{right}"
    return hashlib.sha256(combined.encode()).hexdigest()

# 叶子节点
leaf0 = "f1b2aba4b424b7b879919d3afb7b204fba69309c9f78de3e5fc63f44dab630a1"
leaf1 = "b077afd1363a3e2c8eb9af6c53503cf95776374c45b9bd4afd51df9ec2db9bb1"
leaf2 = "fc2dcdbde03a1c85ea8ab28c84d45c30fa26f6f08da12eb16720e034569e1d0c"
leaf3 = "4089698cf7ada637a2d01602865e15e26cdeba152e8f3184622f27418a8a3317"

# 计算父节点
parent1 = hash_pair(leaf0, leaf1)
parent2 = hash_pair(leaf2, leaf3)
root = hash_pair(parent1, parent2)

print(f"Parent 1 (leaf0 + leaf1): {parent1}")
print(f"Parent 2 (leaf2 + leaf3): {parent2}")
print(f"Root (parent1 + parent2): {root}")

# 验证 Leaf 0 的证明
print("\nVerifying Leaf 0:")
proof_path = [
    (leaf1, False),  # 兄弟是 leaf1，当前是左孩子
    (parent2, False)  # 兄弟是 parent2，当前是左孩子
]

current_hash = leaf0
print(f"Start: {current_hash}")

for sibling_hash, is_right in proof_path:
    if is_right:
        current_hash = hash_pair(sibling_hash, current_hash)
        print(f"Step: hash({sibling_hash} + {current_hash}) = {current_hash}")
    else:
        current_hash = hash_pair(current_hash, sibling_hash)
        print(f"Step: hash({current_hash} + {sibling_hash}) = {current_hash}")

print(f"Final: {current_hash}")
print(f"Expected root: {root}")
print(f"Match: {current_hash == root}")

# 验证 Leaf 1 的证明
print("\nVerifying Leaf 1:")
proof_path1 = [
    (leaf0, True),   # 兄弟是 leaf0，当前是右孩子
    (parent2, False)  # 兄弟是 parent2，当前是左孩子
]

current_hash1 = leaf1
print(f"Start: {current_hash1}")

for sibling_hash, is_right in proof_path1:
    if is_right:
        current_hash1 = hash_pair(sibling_hash, current_hash1)
        print(f"Step: hash({sibling_hash} + {current_hash1}) = {current_hash1}")
    else:
        current_hash1 = hash_pair(current_hash1, sibling_hash)
        print(f"Step: hash({current_hash1} + {sibling_hash}) = {current_hash1}")

print(f"Final: {current_hash1}")
print(f"Expected root: {root}")
print(f"Match: {current_hash1 == root}")
