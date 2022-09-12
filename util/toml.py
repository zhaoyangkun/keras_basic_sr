import os

import toml


def parse_toml(path):
    """解析 toml 文件

    Args:
        path (_type_): toml 文件路径

    Raises:
        FileNotFoundError: 文件不存在错误

    Returns:
        dict: 字典
    """
    # 判断 toml 文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)
