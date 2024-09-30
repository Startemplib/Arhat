import numpy as np  # 用于矩阵操作
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.font_manager import FontProperties
import os
import cv2
from io import BytesIO
import matplotlib.ticker as ticker
import re
import sympy as sp  # 用于符号矩阵

# 设置 Times New Roman 为主要字体，但如果缺失字符，则回退到仿宋或其他字体
plt.rcParams["font.family"] = [
    "Times New Roman",
    "SimSun",
]  # 'SimHei' 是黑体, 'SimSun' 是宋体


def get_desktop_path():
    """获取当前用户桌面的路径"""
    return os.path.join(os.path.expanduser("~"), "Desktop")


def vp(images, ws=(3000, 2300), wp=(600, 100)):
    # 全局变量
    window_name = "Image Viewer"
    constant_ws = ws  # 窗口大小可由参数输入

    # 定义鼠标事件回调函数
    def mouse_callback(event, x, y, flags, param):
        param["mouse_x"], param["mouse_y"] = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                param["zoom_factor"] += 0.37
            else:
                param["zoom_factor"] = max(0.1, param["zoom_factor"] - 0.37)
        elif event == cv2.EVENT_LBUTTONDOWN:
            param["mouse_left_click"] = True

    # 调整图像以保持鼠标为中心的缩放
    def adjust_to_mouse_center(img_cv, zoom_factor, mouse_x, mouse_y):
        h, w, _ = img_cv.shape
        relative_mouse_x = mouse_x / constant_ws[0]
        relative_mouse_y = mouse_y / constant_ws[1]
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        background = np.ones((constant_ws[1], constant_ws[0], 3), dtype=np.uint8) * 255
        resized = cv2.resize(img_cv, (new_w, new_h))

        start_x = max(0, int(relative_mouse_x * new_w - constant_ws[0] // 2))
        start_y = max(0, int(relative_mouse_y * new_h - constant_ws[1] // 2))
        end_x = min(start_x + constant_ws[0], new_w)
        end_y = min(start_y + constant_ws[1], new_h)

        cropped_resized = resized[start_y:end_y, start_x:end_x]
        background[0 : cropped_resized.shape[0], 0 : cropped_resized.shape[1]] = (
            cropped_resized
        )

        return background

    # 显示图像的函数
    def display_image(img_cv, param):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, *wp)  # 使用传入的窗口位置
        cv2.resizeWindow(window_name, *constant_ws)

        while True:
            adjusted_img = adjust_to_mouse_center(
                img_cv, param["zoom_factor"], param["mouse_x"], param["mouse_y"]
            )
            cv2.imshow(window_name, adjusted_img)

            h, w, _ = adjusted_img.shape
            cv2.resizeWindow(window_name, w, h)

            cv2.setMouseCallback(window_name, mouse_callback, param)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 or param["mouse_left_click"]:
                break

        cv2.destroyAllWindows()

    # 初始化鼠标和缩放相关参数
    param = {"zoom_factor": 1.0, "mouse_x": 0, "mouse_y": 0, "mouse_left_click": False}

    # 主逻辑
    for image in images:
        if isinstance(image, str):  # 如果输入是文件路径
            img_cv = cv2.imread(image)
            if img_cv is None:
                print(f"加载图像错误: {image}")
                continue
        else:  # 如果输入是已经加载的图像
            img_cv = image

        param["mouse_left_click"] = False  # 每次展示图片时重置点击状态
        display_image(img_cv, param)


def fic(fig, pad_inches=0.3):
    """将 Matplotlib figure 转换为 OpenCV 可处理的图像，并保留边缘"""
    buf = BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=pad_inches
    )  # 使用 'bbox_inches' 和 'pad_inches'
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, 1)  # 解码为 OpenCV 格式的图像
    buf.close()
    return img_cv


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os


def vnm(
    matrix,
    cs=0.1,
    r=19,
    t=1,
    k=2.1,
    title="Matrix Viewer",
    q=300,
    cmap="viridis",
    path=None,  # 用户可以指定保存路径
    sd=0,  # 增加是否保存到桌面的选项，默认为0不保存
    latex=0,  # 新增参数，控制是否使用LaTeX渲染
):
    rows, cols = matrix.shape
    # 每个格子的基准尺寸
    fig_width = cols * cs
    fig_height = rows * cs
    title_fontsize = max(rows, cols) * 0.3 * t

    # 如果 latex 为 1，则启用 LaTeX 渲染
    if latex == 1:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=q)
    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    # 自动调整字体大小
    font_size = cs * r  # 动态调整字体大小
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

    tick_fontsize = font_size * 3 * k
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 添加颜色条，并设置颜色条的刻度格式
    cbar = fig.colorbar(cax)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # 设置标题，可以使用 LaTeX 格式
    ax.set_title(title, fontdict={"fontsize": title_fontsize})

    img_cv = fic(fig)  # 将 figure 转换为 OpenCV 格式的图像
    vp([img_cv])  # 使用 vp 函数显示图像

    if sd == 1:  # 检查是否要保存到桌面
        desktop_path = get_desktop_path()
        path = os.path.join(desktop_path, f"Matrix-{rows}x{cols}.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"图像已保存到桌面: {path}")
    elif path:  # 如果提供了保存路径
        plt.savefig(path, bbox_inches="tight")
        print(f"图像已保存到: {path}")

    plt.close(fig)  # 关闭图像，避免占用内存


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
    g=5,
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
    scale = max(rows, cols)
    title_fontsize = scale * 0.3 * t

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

    # 添加自动生成的列索引（列号）到矩阵底部
    for j in range(cols):
        ax.text(
            j + 0.5,
            rows + 0.5,  # 将列索引放置在矩阵底部
            str(j + 1),  # 列索引从 1 开始
            va="center",
            ha="center",
            fontsize=font_size * 0.7,  # 调整字体大小
            color="blue",  # 可以选择改变颜色
        )

    # 显示标题，并设置适当的标题位置，使用 `pad` 增加标题与矩阵的间距
    plt.title(
        title, fontsize=title_fontsize, pad=g * scale
    )  # pad 值增加以确保标题与列索引不重叠

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
