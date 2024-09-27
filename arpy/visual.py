import numpy as np  # 用于矩阵操作
import matplotlib.pyplot as plt  # 用于绘图


def vnm(matrix, title, cmap="viridis", title_fontsize=20, dpi=300):
    rows, cols = matrix.shape
    cell_size = 0.3  # 每个格子的基准尺寸
    fig_width = cols * cell_size
    fig_height = rows * cell_size

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    # 自动调整字体大小
    font_size = cell_size * 17  # 动态调整字体大小
    for i in range(rows):
        for j in range(cols):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=font_size,
            )

    # 添加格子的黑色边框
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # 隐藏默认的主轴刻度
    ax.tick_params(which="minor", size=0)

    fig.colorbar(cax)
    ax.set_title(title, fontsize=title_fontsize)
    plt.show()
