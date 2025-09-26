import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import cv2
from io import BytesIO
import matplotlib.ticker as ticker
import re
import sympy as sp


####### generate a tridiagonal matrix #######

### upper                 (list)       上对角线元素列表，长度为 n-1
### diagonal              (list)       主对角线元素列表，长度为 n
### lower                 (list)       下对角线元素列表，长度为 n-1
### latex                 (int)        是否使用 LaTeX 渲染输出（1 表示使用，0 表示否），默认值为0


def gtdm(upper, diagonal, lower, latex=0):
    n = len(diagonal)

    # 检查输入长度是否合理
    if len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("下对角线和上对角线的长度必须为 n - 1")

    # 如果 latex=1，生成 LaTeX 符号列表
    if latex == 1:
        matrix_latex = [["0" for _ in range(n)] for _ in range(n)]

        for i in range(n):
            if isinstance(diagonal[i], (int, float)):  # 检查是否为数字
                matrix_latex[i][i] = f"${diagonal[i]}$"  # 主对角线元素
            else:
                matrix_latex[i][i] = f"${diagonal[i]}$"

        for i in range(1, n):
            if isinstance(upper[i - 1], (int, float)):  # 检查是否为数字
                matrix_latex[i - 1][i] = f"${upper[i - 1]}$"  # 上对角线元素
            else:
                matrix_latex[i - 1][i] = f"${upper[i - 1]}$"

            if isinstance(lower[i - 1], (int, float)):  # 检查是否为数字
                matrix_latex[i][i - 1] = f"${lower[i - 1]}$"  # 下对角线元素
            else:
                matrix_latex[i][i - 1] = f"${lower[i - 1]}$"

        # 添加省略号符号
        matrix_latex.append([r"$\vdots$" for _ in range(n)])
        matrix_latex[-1][-1] = r"$\ddots$"

        return matrix_latex

    else:
        matrix_sympy = sp.zeros(n)  # 创建 n x n 的零矩阵

        for i in range(n):
            if isinstance(diagonal[i], (int, float)):  # 如果是数字，直接赋值
                matrix_sympy[i, i] = diagonal[i]  # 主对角线元素
            else:
                matrix_sympy[i, i] = (
                    sp.symbols(diagonal[i])
                    if not isinstance(diagonal[i], sp.Basic)
                    else diagonal[i]
                )

        for i in range(1, n):
            if isinstance(lower[i - 1], (int, float)):  # 如果是数字，直接赋值
                matrix_sympy[i, i - 1] = lower[i - 1]  # 下对角线元素
            else:
                matrix_sympy[i, i - 1] = (
                    sp.symbols(lower[i - 1])
                    if not isinstance(lower[i - 1], sp.Basic)
                    else lower[i - 1]
                )

            if isinstance(upper[i - 1], (int, float)):  # 如果是数字，直接赋值
                matrix_sympy[i - 1, i] = upper[i - 1]  # 上对角线元素
            else:
                matrix_sympy[i - 1, i] = (
                    sp.symbols(upper[i - 1])
                    if not isinstance(upper[i - 1], sp.Basic)
                    else upper[i - 1]
                )

        return matrix_sympy
