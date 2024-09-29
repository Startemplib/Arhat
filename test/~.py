from arpy.visual import *


def vsm(
    matrix,
    cs=1,
    r=19,
    t=30,
    k=2.1,
    title="Symbol Matrix",
    q=300,
    path=None,  # 用户可以指定保存路径
    sd=0,  # 增加是否保存到桌面的选项，默认为0不保存
    latex=1,  # 使用LaTeX来渲染符号矩阵，默认开启
):
    # 如果传递的是 SymPy 矩阵或者 NumPy 数组，可以直接使用 shape
    if hasattr(matrix, "shape"):
        rows, cols = matrix.shape
        sympy_matrix = True
    # 如果传递的是 Python 列表，使用 len 来确定行列数
    elif isinstance(matrix, list):
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        sympy_matrix = False
    else:
        raise ValueError("Unsupported matrix type")

    # 每个格子的基准尺寸
    fig_width = cols * cs
    fig_height = rows * cs
    title_fontsize = max(rows, cols) * 0.3 * t

    # 启用 LaTeX 渲染
    if latex == 1:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=q)

    # 设置网格线，只在矩阵内部显示，不延伸到黑色边框之外
    ax.set_xticks(np.arange(0, cols + 1), minor=False)
    ax.set_yticks(np.arange(0, rows + 1), minor=False)
    ax.grid(which="major", color="gray", linestyle="-", linewidth=1)

    # 仅在矩阵单元格内部绘制网格线
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    # 添加矩形黑色边框，确保它包围矩阵，不与灰色网格线冲突
    rect = plt.Rectangle((0, 0), cols, rows, fill=False, color="black", linewidth=2)
    ax.add_patch(rect)

    # 自动调整字体大小
    font_size = cs * r  # 动态调整字体大小
    for i in range(rows):
        for j in range(cols):
            if sympy_matrix:
                # 使用 matrix[i, j] 访问 SymPy 矩阵中的元素
                symbol_str = sp.latex(matrix[i, j])
            else:
                # 对于列表中的字符串，直接使用
                symbol_str = matrix[i][j]
            ax.text(
                j + 0.5,  # 将文本居中放置
                i + 0.5,
                f"${symbol_str}$",  # 使用 LaTeX 语法显示符号
                ha="center",
                va="center",
                color="black",
                fontsize=font_size,
            )

    # 隐藏刻度标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 添加自动生成的行索引（行号）
    for i in range(rows):
        ax.text(
            -0.5,  # 将行索引放置在左侧
            i + 0.5,
            str(i + 1),  # 行索引从 1 开始
            va="center",
            ha="right",
            fontsize=font_size * 0.7,  # 调整字体大小
            color="blue",  # 可以选择改变颜色
        )

    # 添加自动生成的列索引（列号）
    for j in range(cols):
        ax.text(
            j + 0.5,
            -0.5,  # 将列索引放置在顶部
            str(j + 1),  # 列索引从 1 开始
            va="top",
            ha="center",
            fontsize=font_size * 0.7,  # 调整字体大小
            color="blue",  # 可以选择改变颜色
        )

    # 显示标题，并设置适当的标题位置
    plt.title(title, fontsize=title_fontsize, pad=20)

    img_cv = fic(fig)  # 将 figure 转换为 OpenCV 格式的图像
    vp([img_cv])  # 使用 vp 函数显示图像

    if sd == 1:  # 检查是否要保存到桌面
        desktop_path = get_desktop_path()
        path = os.path.join(desktop_path, f"Symbol_Matrix-{rows}x{cols}.png")
        plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
        print(f"图像已保存到桌面: {path}")
    elif path:  # 如果提供了保存路径
        plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
        print(f"图像已保存到: {path}")

    plt.close(fig)  # 关闭图像，避免占用内存


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
