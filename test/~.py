from arpy.visual import *


# 示例：创建符号矩阵
x, y = sp.symbols("x y")  # 定义符号
matrix = sp.Matrix([[x + y, x - y], [x * y, x / y]])  # 2x2 符号矩阵
matrix_latex = [
    ["0", r"$v_{n-1}$", "0", "0", "0", "0", r"$\cdots$"],
    [r"$v_{n-1}$", "0", r"$\kappa_{n-2}$", "0", "0", "0", r"$\cdots$"],
    ["0", r"$\kappa_{n-2}$", "0", r"$v_{n-2}$", "0", "0", r"$\cdots$"],
    ["0", "0", r"$v_{n-2}$", "0", r"$\kappa_{n-3}$", "0", r"$\cdots$"],
    ["0", "0", "0", r"$\kappa_{n-3}$", "0", r"$v_{n-3}$", r"$\cdots$"],
    [
        r"$\vdots$",
        r"$\vdots$",
        r"$\vdots$",
        r"$\vdots$",
        r"$\vdots$",
        r"$\vdots$",
        r"$\ddots$",
    ],
]

# 符号矩阵显示
vsm(matrix, latex=0, title="我", q=300)

# 使用 LaTeX 矩阵显示
vsm(matrix_latex, latex=1, title=r"$v_{n-1}$", q=300, sd=1)
