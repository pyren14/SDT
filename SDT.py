import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

# 加载 pkl 文件和 joblib 文件
with open('soft_decision_tree_16_mnist.pkl', 'rb') as f:
    tree_pkl = pickle.load(f)

node_path_left_right = tree_pkl['node_path_left_right']  # 路径方向矩阵
leaf_depths = tree_pkl['leaf_depth']  # 每条路径深度

clf = joblib.load('16_rf.joblib')
tree = clf.estimators_[0].tree_

features = tree.feature  # 节点特征索引
thresholds = tree.threshold  # 节点分裂阈值
values = tree.value  # 叶子节点对应分类分布


# 格式化单条路径
def format_path(path_idx):
    path_text = ""
    depth = int(leaf_depths[path_idx])  # 获取当前路径的深度

    for d in range(depth):
        node_idx = node_path_left_right[path_idx, d, 0]  # 节点索引
        direction = node_path_left_right[path_idx, d, 1]  # 方向

        if features[node_idx] == -2:  # 叶子节点
            class_id = np.argmax(values[node_idx])  # 取最大概率对应的分类
            path_text += f"|   |--- class: {class_id}\n"
            return path_text

        # 添加节点分裂信息
        if direction == 0:  # 向左
            path_text += f"|   |--- feature_{features[node_idx]} <= {thresholds[node_idx]:.10f}\n"
        elif direction == 1:  # 向右
            path_text += f"|   |   > feature_{features[node_idx]} > {thresholds[node_idx]:.10f}\n"
        else:
            print(f"Warning: Invalid direction {direction} for node {node_idx}, path {path_idx}")

    # 如果路径未正常结束，补充分类信息
    if features[node_idx] != -2:
        class_id = np.argmax(values[node_idx])
        path_text += f"|   |--- class: {class_id}\n"

    return path_text


# 生成完整树文本
def generate_tree_text():
    tree_text = ""
    for path_idx in range(len(leaf_depths)):
        tree_text += f"Path {path_idx + 1}:\n"  # 添加路径编号
        tree_text += format_path(path_idx)
    return tree_text


# 主函数
if __name__ == "__main__":
    try:
        print("开始生成树结构...")
        tree_text = generate_tree_text()

        # 保存到文件
        output_file = "SDT_fixed.txt"
        with open(output_file, "w") as f:
            f.write(tree_text)

        print(f"树结构已生成并保存到 {output_file}")
    except Exception as e:
        print(f"Error: {e}")
