from arpy.visual import *
from arpy.matrix import *

# 示例：创建符号矩阵
x, y = sp.symbols("x y")  # 定义符号
matrix = sp.Matrix([[x + y, x - y], [x * y, x / y]])  # 2x2 符号矩阵

# print(type(matrix))

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


# print(type(matrix_latex))
# 符号矩阵显示
# vsm(matrix, latex=0, title="我", q=300)

# 使用 LaTeX 矩阵显示
# vsm(matrix_latex, latex=1, title=r"$v_{n-1}$", q=300, sd=1)

# 示例用法
diagonal = ["0", "0", "0", "0", "0"]  # 主对角线元素
lower = [f"v_{{n-{i}}}" for i in range(1, 5)]  # 下对角线元素
upper = [f"\\kappa_{{n-{i}}}" for i in range(1, 5)]  # 上对角线元素

# LaTeX 格式输出
latex_matrix = gtdm(upper, diagonal, lower, latex=1)
# print(latex_matrix)
# print(type(latex_matrix))

# vsm(latex_matrix, latex=1, title=r"$v_{n-1}$", q=300, sd=1)


diagonal = [x, y, y, x]  # 主对角线元素
lower = [y, x, y]  # 下对角线元素
upper = [y, x, y]  # 上对角线元素

# SymPy 矩阵输出
sympy_matrix = gtdm(upper, diagonal, lower, latex=0)
print(sympy_matrix)
print(type(sympy_matrix))
vsm(sympy_matrix, latex=0, title="我", q=300, sd=1)
