import matplotlib.pyplot as plt
import numpy as np

# 启用 LaTeX 渲染
plt.rcParams["text.usetex"] = True

# 创建一个 SSH 模型相关的矩阵，手动指定 LaTeX 符号
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

# 矩阵的大小
rows = len(matrix_latex)
cols = len(matrix_latex[0])

# 创建图和子图
fig, ax = plt.subplots()

# 创建矩阵热图的背景 (显示颜色)
ax.matshow(np.zeros((rows, cols)), cmap="coolwarm")

# 设置动态字体大小
cs = 12  # 字体大小系数
r = 1  # 调整倍率
font_size = cs * r

# 在每个单元格中填充 LaTeX 矩阵元素
for i in range(rows):
    for j in range(cols):
        ax.text(
            j,
            i,
            matrix_latex[i][j],
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
        )

# 隐藏坐标轴
ax.set_xticks([])
ax.set_yticks([])

# 显示图
plt.show()
