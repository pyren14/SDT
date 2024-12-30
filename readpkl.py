import pickle
import pandas as pd

# 指定 .pkl 文件路径和输出 .csv 文件路径
pkl_file_path = "soft_decision_tree_16_mnist.pkl"
csv_file_path = "soft_decision_tree_full.csv"

# 读取 .pkl 文件
with open(pkl_file_path, "rb") as f:
    tree_pkl = pickle.load(f)

# 检查数据类型并转换为 DataFrame
if isinstance(tree_pkl, pd.DataFrame):
    # 如果已经是 DataFrame
    df = tree_pkl
elif isinstance(tree_pkl, dict):
    # 如果是字典，将其转换为 DataFrame
    df = pd.DataFrame([tree_pkl])
elif isinstance(tree_pkl, list):
    # 如果是列表，将其转换为 DataFrame
    df = pd.DataFrame(tree_pkl)
else:
    raise ValueError("不支持的 .pkl 文件数据类型，无法直接转换为 CSV。")

# 保存为 .csv 文件
df.to_csv(csv_file_path, index=False, encoding="utf-8")
print(f"数据已完整保存到: {csv_file_path}")
