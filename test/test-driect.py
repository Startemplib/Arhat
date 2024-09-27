import sys

from arpy.visual import *

# 使用 arpy 模块中的 vnm 函数
matrix = np.array([[1, 2], [3, 4]])  # 将列表转换为 NumPy 数组
vnm(matrix, "Test Matrix")
