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


print(type(matrix_latex))
# 符号矩阵显示
# vsm(matrix, latex=0, title="我", q=300)

# 使用 LaTeX 矩阵显示
# vsm(matrix_latex, latex=1, title=r"$v_{n-1}$", q=300, sd=1)


def generate_tridiagonal_matrix(diagonal, lower, upper):
    n = len(diagonal)
    if len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("下对角线和上对角线的长度必须为 n - 1")

    matrix = sp.zeros(n)  # 创建 n x n 的零矩阵

    for i in range(n):
        matrix[i, i] = diagonal[i]  # 主对角线元素
        if i > 0:
            matrix[i, i - 1] = lower[i - 1]  # 下对角线元素
        if i < n - 1:
            matrix[i, i + 1] = upper[i]  # 上对角线元素

    return matrix


n = 10
# 示例用法
diagonal = list(range(n))  # 正确的用法，生成从0到n-1的列表
lower = list(range(n, 2 * n - 1))  # 正确的用法，生成从n到2n-2的列表
upper = list(range(2 * n, 3 * n - 1))  # 正确的用法，生成从2n到3n-2的列表


tridiagonal_matrix = generate_tridiagonal_matrix(diagonal, lower, upper)
print(tridiagonal_matrix)

vsm(tridiagonal_matrix, latex=0, title="我", q=300)


def generate_tridiagonal_matrix(diagonal, lower, upper, latex=0):
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

    # 否则，生成 SymPy 矩阵
    else:
        matrix_sympy = sp.zeros(n)  # 创建 n x n 的零矩阵

        for i in range(n):
            matrix_sympy[i, i] = sp.symbols(diagonal[i])  # 主对角线元素
        for i in range(1, n):
            matrix_sympy[i, i - 1] = sp.symbols(lower[i - 1])  # 下对角线元素
            matrix_sympy[i - 1, i] = sp.symbols(upper[i - 1])  # 上对角线元素

        return matrix_sympy


# 示例用法
diagonal = ["0", "0", "0", "0", "0"]  # 主对角线元素
lower = [f"v_{{n-{i}}}" for i in range(1, 5)]  # 下对角线元素
upper = [f"\\kappa_{{n-{i}}}" for i in range(1, 5)]  # 上对角线元素

# LaTeX 格式输出
latex_matrix = generate_tridiagonal_matrix(diagonal, lower, upper, latex=1)
print(latex_matrix)
print(type(latex_matrix))

vsm(latex_matrix, latex=1, title=r"$v_{n-1}$", q=300, sd=1)

# SymPy 矩阵输出
sympy_matrix = generate_tridiagonal_matrix(diagonal, lower, upper, latex=0)
print(sympy_matrix)
print(type(sympy_matrix))
vsm(sympy_matrix, latex=0, title="我", q=300)
