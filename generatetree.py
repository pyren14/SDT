class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, class_id=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_id = class_id

    def is_leaf(self):
        return self.class_id is not None


def parse_paths_from_file(file_path):
    """
    Parse the paths from the input file.
    """
    paths = []
    current_path = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Path"):  # Skip empty lines and path headers
                if current_path:
                    paths.append(current_path)
                current_path = []
                continue

            if "|--- class:" in line:
                # Parse the class ID
                class_id = int(line.split(":")[1].strip())
                current_path.append(("class", class_id))
            elif "|--- feature_" in line or "|   > feature_" in line:
                # Parse feature, threshold, and direction
                parts = line.split()
                feature = int(parts[1].split("_")[1])
                threshold = float(parts[3])
                direction = "left" if "<=" in parts[2] else "right"
                current_path.append((feature, threshold, direction))
            else:
                print(f"Warning: Unexpected line format: {line}")

        if current_path:
            paths.append(current_path)

    return paths


def build_tree_from_paths(paths):
    """
    Build a decision tree from the parsed paths.
    """
    root = TreeNode()

    for path in paths:
        current_node = root
        for node in path:
            if node[0] == "class":
                # Assign class ID to the leaf node
                current_node.class_id = node[1]
            else:
                feature, threshold, direction = node
                if direction == "left":
                    if not current_node.left:
                        current_node.left = TreeNode(feature=feature, threshold=threshold)
                    current_node = current_node.left
                elif direction == "right":
                    if not current_node.right:
                        current_node.right = TreeNode(feature=feature, threshold=threshold)
                    current_node = current_node.right
                else:
                    print(f"Warning: Invalid direction '{direction}' in path")

    return root


def format_tree(node, indent=""):
    """
    Format the tree into readable text.
    """
    if node.is_leaf():
        return f"{indent}|--- class: {node.class_id}\n"
    result = f"{indent}|--- feature_{node.feature} <= {node.threshold:.10f}\n"
    if node.left:
        result += format_tree(node.left, indent + "|   ")
    if node.right:
        result += format_tree(node.right, indent + "|   > ")
    return result


if __name__ == "__main__":
    input_file = "SDT_fixed.txt"  # Replace with your input file name
    output_file = "SDT_combined_tree.txt"

    print("开始从文件中读取路径并构建树...")
    paths = parse_paths_from_file(input_file)
    tree_root = build_tree_from_paths(paths)
    tree_text = format_tree(tree_root)

    with open(output_file, "w") as f:
        f.write(tree_text)

    print(f"树结构已保存到 {output_file}")
