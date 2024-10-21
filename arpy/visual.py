################################################### 可视化函数库 ###################################################

# General Libraries
import os  # For operating system interactions
import re  # For regular expressions
import io  # For input/output operations
import numpy as np  # For matrix operations and numerical computations

# Mathematical and Symbolic Libraries
import sympy as sp  # For symbolic mathematics

# Image Processing Libraries
import cv2  # For image processing (OpenCV)
from PIL import Image  # For handling images

# Plotting Libraries
import matplotlib as mpl  # For overall matplotlib settings
import matplotlib.pyplot as plt  # For plotting
from matplotlib.ticker import FuncFormatter  # For custom tick formatting
import matplotlib.ticker as ticker  # For additional tick customization
from matplotlib.font_manager import FontProperties  # For handling fonts

# Display Libraries (IPython-specific)
from IPython.display import (
    display,
    Image as IPImage,
)  # For displaying images in notebooks


# Additional Libraries
from io import BytesIO  # For handling byte streams
import ctypes  # For calling C functions and other low-level operations

####### 字体设置(font settings) #######

plt.rcParams["font.family"] = [
    "Times New Roman",
    "SimSun",
]  # 'SimHei' 是黑体, 'SimSun' 是宋体


####### 桌面地址抓取(desktop address capture) #######


def get_desktop_path():
    """获取当前用户桌面的路径"""
    return os.path.join(os.path.expanduser("~"), "Desktop")


####### 图片观察(view photo) #######

### 鼠标左键与enter键退出图片观察
### 图片方位自动随鼠标位置移动
### 滑轮滑动缩放

### images                 (str or fig or list)     图片地址or图片文件夹地址or图片变量or图片地址组成的list变量or图片变量所组成的list变量
### ws (window_size)       (tuple)          窗口大小，默认值为 (3000, 2300)
### wp (window_position)   (tuple)          窗口在屏幕上的位置，默认值为 (600, 100)


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

    if isinstance(images, str) or not isinstance(images, list):
        images = [images]  # 如果是单个路径或图像，转换为单元素列表

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


####### 图片格式转换(figure image convertor) #######

### fig                   (Figure)     Matplotlib 的 figure 对象
### pi (pad_inches)       (float)      图像四周的内边距，单位为英寸，默认值为1
### form (format)         (string)     转换结果类型指定

### return:
### img_cv                (ndarray)    OpenCV 格式的图像数组


def fic(fig, form="png", pi=1):
    buf = BytesIO()
    fig.savefig(buf, format=form, bbox_inches="tight", pad_inches=pi * 0.3)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, 1)
    buf.close()
    return img_cv  # 返回 OpenCV 格式的图像


####### 可视化数值矩阵(view numerical matrix) #######

### matrix                               (ndarray)    待展示的矩阵数据，n行m列的二维数组
### cs (cell_size)                       (int)        每个格子的尺寸系数，默认值为1
### f (font_size)                        (int)        字体尺寸系数，默认值为1
### t (title_size)                       (int)        标题尺寸系数，默认值为1
### k (tick_size)                        (int)        坐标刻度字体大小系数，默认值为1
### cbs (font_size of color bar)          int)        彩色颜色条刻度字体大小系数，默认值为1
### title                                (str)        矩阵的标题，默认值为 "Numerical-Matrix"
### q (quality)                          (int)        图像质量(DPI)，默认值为300
### cmap (colormap)                      (str)        颜色映射表，默认值为 "viridis"
### path (save_path)                     (str)        可选参数，保存路径，默认值为 None
### sd (save_desktop)                    (int)        是否保存到桌面（1 表示是，0 表示否），默认值为0
### latex (use_latex)                    (int)        是否启用LaTeX渲染，1 表示启用，0 表示禁用，默认值为0
### sxt(spacing of the x-axis ticks)     (int)        x轴刻度间距
### yxt(spacing of the y-axis ticks)     (int)        y轴刻度间距


def vnm(
    matrix,
    cs=1,
    f=1,
    t=1,
    k=1,
    title="Numerical-Matrix",
    q=300,
    cmap="viridis",
    path=None,  # 用户可以指定保存路径
    sd=0,  # 增加是否保存到桌面的选项，默认为0不保存
    latex=0,  # 新增参数，控制是否使用LaTeX渲染
    sxt=1,
    syt=1,
    cbs=1,
):
    rows, cols = matrix.shape
    # 每个格子的基准尺寸
    fig_width = cols * cs * 0.1
    fig_height = rows * cs * 0.1
    title_fontsize = max(rows, cols) * 0.3 * t

    # 如果 latex 为 1，则启用 LaTeX 渲染
    if latex == 1:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=q)

    abs_matrix = np.abs(matrix)  # Get the magnitude of the complex matrix
    cax = ax.imshow(abs_matrix, cmap=cmap, interpolation="nearest")

    # 自动调整字体大小
    font_size = cs * f * 19  # 动态调整字体大小
    for i in range(rows):
        for j in range(cols):
            real_part = matrix[i, j].real
            imag_part = matrix[i, j].imag
            if imag_part >= 0:
                ax.text(
                    j,
                    i,
                    f"{real_part:.2f}+{imag_part:.2f}i",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=font_size,
                )
            else:
                ax.text(
                    j,
                    i,
                    f"{real_part:.2f}{imag_part:.2f}i",  # Handle negative imaginary part
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=font_size,
                )

    tick_fontsize = font_size * 6 * k
    ax.set_xticks(
        np.arange(0, cols, step=sxt)
    )  # Control the tick positions on the x-axis
    ax.set_yticks(
        np.arange(0, rows, step=syt)
    )  # Control the tick positions on the y-axis
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 添加颜色条，并设置颜色条的刻度格式
    cbar = fig.colorbar(cax)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    cbar.ax.tick_params(labelsize=cbs * tick_fontsize)

    # 设置标题，可以使用 LaTeX 格式
    ax.set_title(title, fontdict={"fontsize": title_fontsize})

    img_cv = fic(fig)  # 将 figure 转换为 OpenCV 格式的图像
    vp([img_cv])  # 使用 vp 函数显示图像

    try:
        if path and sd == 1:  # 如果提供了保存路径且 sd == 1，优先使用 path 并提示信息
            plt.savefig(path, bbox_inches="tight")
            print(f"图像已保存到指定路径: {path}，尽管 sd == 1，优先使用了自定义路径。")
        elif path:  # 仅提供了保存路径时，直接使用 path
            plt.savefig(path, bbox_inches="tight")
            print(f"图像已保存到指定路径: {path}")
        elif sd == 1:  # 如果未提供 path 且 sd == 1，则保存到桌面
            desktop_path = get_desktop_path()
            path = os.path.join(desktop_path, f"Matrix-{rows}x{cols}.png")
            plt.savefig(path, bbox_inches="tight")
            print(f"图像已保存到桌面: {path}")
        else:
            print("没有提供保存路径，图像未保存。")
    except Exception as e:
        print(f"保存图像时发生错误: {e}")

    plt.close(fig)  # 关闭图像，避免占用内存


####### 可视化符号矩阵(view Symbolic matrix) #######

### matrix                (ndarray)    待展示的符号矩阵数据，n行m列的二维数组
### cs (cell_size)        (int)        每个格子的尺寸系数，默认值为1
### f (font_size)         (int)        字体尺寸系数，默认值为1
### t (title_size)        (int)        标题尺寸系数，默认值为1
### title                 (str)        符号矩阵的标题，默认值为 "Symbolic-Matrix"
### q (quality)           (int)        图像质量(DPI)，默认值为300
### path (save_path)      (str)        可选参数，保存路径，默认值为 None
### sd (save_desktop)     (int)        是否保存到桌面（1 表示是，0 表示否），默认值为0
### latex (use_latex)     (int)        使用LaTeX渲染符号矩阵，默认值为1
### tg  (title_grid)      (int)        控制title到矩阵上边grid的留空距离，默认值为1


def vsm(
    matrix,
    cs=1,
    f=1,
    t=1,
    title="Symbolic-Matrix",
    q=300,
    path=None,  # 用户可以指定保存路径
    sd=0,  # 增加是否保存到桌面的选项，默认为0不保存
    latex=1,  # 使用LaTeX来渲染符号矩阵，默认开启
    tg=1,
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
    title_fontsize = scale * 9 * t

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
    font_size = cs * 19 * f  # 动态调整字体大小
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
        title, fontsize=title_fontsize, pad=tg * 5 * scale
    )  # pad 值增加以确保标题与列索引不重叠

    img_cv = fic(fig)  # 将 figure 转换为 OpenCV 格式的图像
    vp([img_cv])  # 使用 vp 函数显示图像

    try:
        if path and sd == 1:  # 如果提供了保存路径且 sd == 1，优先使用 path 并提示信息
            plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
            print(f"图像已保存到指定路径: {path}，尽管 sd == 1，优先使用了自定义路径。")
        elif path:  # 仅提供了保存路径时，直接使用 path
            plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
            print(f"图像已保存到指定路径: {path}")
        elif sd == 1:  # 如果未提供 path 且 sd == 1，则保存到桌面
            desktop_path = get_desktop_path()
            path = os.path.join(desktop_path, f"Symbol_Matrix-{rows}x{cols}.png")
            plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
            print(f"图像已保存到桌面: {path}")
        else:
            print("没有提供保存路径，图像未保存。")
    except Exception as e:
        print(f"保存图像时发生错误: {e}")

    plt.close(fig)  # 关闭图像，避免占用内存


####### 绘图plot(LaTeX + matplotlib) #######

### x_data            (ndarray)   x轴数据, 作为绘图输入的数值数组
### y_data            (ndarray)   y轴数据, 作为绘图输入的数值数组
### title             (str)       图表的标题, 描述图像内容，支持LaTeX
### xlabel            (str)       x轴的标签, 描述数据含义，支持LaTeX
### ylabel            (str)       y轴的标签, 描述数据含义，支持LaTeX
### ts(title size)    (int)       图表标题的尺寸
### xs(xlabel size)   (int)       x轴标签的尺寸
### ys(ylabel size)   (int)       y轴标签的尺寸
### z (zoom factor)   (float)     缩放因子, 用于调整图片的尺寸, 默认值为0.3
### s (figure size)   (tuple)     图像的尺寸, 以元组形式表示 (宽, 高), 默认为 (8, 4)
### dpi (quality)     (int)       图像质量(DPI), 默认值为300, 影响输出图像的分辨率


def plot(
    x_data, y_data, xlabel, ylabel, title, z=0.3, s=(7, 5), dpi=300, ts=16, xs=11, ys=11
):

    # Store the current backend
    original_backend = mpl.get_backend()

    # Switch to 'pgf' backend to render with LaTeX
    mpl.use("pgf")

    # Set up the configuration for LaTeX and fonts
    pgf_with_latex = {
        "pgf.rcfonts": False,  # Do not use default matplotlib fonts, use the one specified in font.family
        "text.usetex": True,  # Use LaTeX to write all text
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Times New Roman"],  # Use Times New Roman for English text
        "pgf.preamble": r"\usepackage{xeCJK}\setCJKmainfont{KaiTi} \usepackage{amsmath}",
    }

    mpl.rcParams.update(pgf_with_latex)

    # Create the plot
    plt.figure(figsize=s)  # Set figure size
    plt.plot(x_data, y_data)

    # Set labels and title with specified font sizes
    plt.xlabel(xlabel, fontsize=xs)
    plt.ylabel(ylabel, fontsize=ys)
    plt.title(title, fontsize=ts)

    plt.grid(True)

    # Disable scientific notation (which causes asterisks in ticks)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

    plt.tight_layout()

    # Create an in-memory binary stream to store the image
    img_buffer = io.BytesIO()

    # Save the figure to this in-memory buffer as a PNG with DPI setting for quality control
    plt.savefig(
        img_buffer, format="png", dpi=dpi
    )  # Set dpi for higher quality (300 DPI is a common high-res setting)

    plt.close("all")

    # Reset the buffer's position to the start
    img_buffer.seek(0)

    # Read the image to get its original dimensions
    image = Image.open(img_buffer)
    original_width, original_height = image.size

    # Calculate the new dimensions based on the zoom factor
    new_width = int(original_width * z)
    new_height = int(original_height * z)

    # Display the image with the resized dimensions
    display(IPImage(data=img_buffer.getvalue(), width=new_width, height=new_height))

    # Close the buffer when done (optional cleanup)
    img_buffer.close()

    # Restore the original backend
    mpl.use(original_backend)


####### 号角Horn #######

### text                       (str)
### w(window_width)            (int)
### h(window_height)           (int)
### t(display_duration/ms)        (int)


def Horn(text, w=800, h=300, t=1370):
    # Adjust figure size to fit the text size (narrow rectangle)
    fig = plt.figure(figsize=(4.53, 1.3137), facecolor="black")

    # Create a plot and add the text in white with Times New Roman italic
    plt.text(
        0.5,
        0.4,
        text,
        fontsize=12,
        fontfamily="Times New Roman",
        fontstyle="italic",
        color="white",
        ha="center",
        va="center",
    )

    # Hide the axes
    plt.gca().set_axis_off()

    # Save the plot to a BytesIO buffer instead of displaying it
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Read the image from the buffer as a numpy array
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert RGB to BGR for OpenCV (since OpenCV uses BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create an OpenCV window
    cv2.namedWindow("Horn", cv2.WINDOW_NORMAL)

    # Get screen dimensions using ctypes
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    # Resize the window to the custom size
    cv2.resizeWindow("Horn", w, h)

    # Calculate window position to center the resized window on the screen
    window_x = (screen_width - w) // 2
    window_y = (screen_height - h) // 2

    # Position the window in the center of the screen
    cv2.moveWindow("Horn", window_x, window_y)
    cv2.setWindowProperty("Horn", cv2.WND_PROP_TOPMOST, 1)

    # Display the image in the OpenCV window
    cv2.imshow("Horn", image)

    # Wait for the specified duration (in milliseconds)
    cv2.waitKey(t)

    # Close the OpenCV window
    cv2.destroyAllWindows()

    # Clean up Matplotlib resources
    plt.close()
