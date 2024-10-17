from arpy import visual

import numpy as np

# 使用 arpy 模块中的 vnm 函数
matrix = np.array([[1, 2], [3, 4]])  # 将列表转换为 NumPy 数组
visual.vnm(matrix, sd=1, path=r"C:\Users\lj\Desktop\Matrix-2x2.svg")
