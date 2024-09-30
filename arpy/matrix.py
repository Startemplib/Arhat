import numpy as np  # 用于矩阵操作
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.font_manager import FontProperties
import os
import cv2
from io import BytesIO
import matplotlib.ticker as ticker
import re
import sympy as sp  # 用于符号矩阵


def gtdm(diagonal, lower, upper, latex=0):
    n = len(diagonal)

    # 检查输入长度是否合理
    if len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("下对角线和上对角线的长度必须为 n - 1")

    # 如果latex=1，生成 LaTeX 符号列表
    if latex == 1:
        matrix_latex = [["0" for _ in range(n)] for _ in range(n)]

        for i in range(n):
            matrix_latex[i][i] = f"${diagonal[i]}$"  # 主对角线元素
        for i in range(1, n):
            matrix_latex[i - 1][i] = f"${upper[i - 1]}$"  # 上对角线元素
            matrix_latex[i][i - 1] = f"${lower[i - 1]}$"  # 下对角线元素

        # 添加省略号符号
        matrix_latex.append([r"$\vdots$" for _ in range(n)])
        matrix_latex[-1][-1] = r"$\ddots$"

        return matrix_latex

    else:
        matrix_sympy = sp.zeros(n)  # 创建 n x n 的零矩阵

        for i in range(n):
            if not isinstance(diagonal[i], sp.Basic):  # 检查是否已经是 SymPy 符号
                matrix_sympy[i, i] = sp.symbols(diagonal[i])  # 主对角线元素
            else:
                matrix_sympy[i, i] = diagonal[i]

        for i in range(1, n):
            if not isinstance(lower[i - 1], sp.Basic):  # 检查是否已经是 SymPy 符号
                matrix_sympy[i, i - 1] = sp.symbols(lower[i - 1])  # 下对角线元素
            else:
                matrix_sympy[i, i - 1] = lower[i - 1]

            if not isinstance(upper[i - 1], sp.Basic):  # 检查是否已经是 SymPy 符号
                matrix_sympy[i - 1, i] = sp.symbols(upper[i - 1])  # 上对角线元素
            else:
                matrix_sympy[i - 1, i] = upper[i - 1]

    return matrix_sympy
