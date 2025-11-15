################################################### Visual ###################################################

import subprocess
from pathlib import Path
import re
import shutil
import tempfile

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
import qrcode

####### font setting #######

plt.rcParams["font.family"] = [
    "Times New Roman",
    "SimSun",
]

####### View Photo (vp) #######

### Left mouse click or Enter key to exit image view
### Image pan auto-follows mouse position
### Mouse wheel to zoom in/out

### images                 (str or fig or list)     Image file path, or loaded image object, or a list of them
### ws (window_size)       (tuple)                  Window size, default is (3000, 2300)
### wp (window_position)   (tuple)                  Window position on screen, default is (600, 100)
### auto                   (bool)                   Enable auto window sizing/positioning, default is False


def vp(images, auto=1, ws=(3000, 2300), wp=(600, 100)):

    def mouse_callback(event, x, y, flags, param):
        param["mouse_x"], param["mouse_y"] = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                param["zoom_factor"] += 0.17
            else:
                param["zoom_factor"] = max(0.1, param["zoom_factor"] - 0.17)
        elif event == cv2.EVENT_LBUTTONDOWN:
            param["mouse_left_click"] = True

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

    def display_image(img_cv, param):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, *wp)
        cv2.resizeWindow(window_name, *constant_ws)

        while True:
            adjusted_img = adjust_to_mouse_center(
                img_cv, param["zoom_factor"], param["mouse_x"], param["mouse_y"]
            )
            cv2.imshow(window_name, adjusted_img)

            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

            h, w, _ = adjusted_img.shape
            cv2.resizeWindow(window_name, w, h)

            cv2.setMouseCallback(window_name, mouse_callback, param)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 or param["mouse_left_click"]:
                break

        cv2.destroyAllWindows()

    param = {"zoom_factor": 1, "mouse_x": 0, "mouse_y": 0, "mouse_left_click": False}

    if isinstance(images, str) or not isinstance(images, list):
        images = [images]

    for image in images:
        if isinstance(image, str):
            img_cv = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                print(f"加载图像错误: {image}")
                continue
        else:
            img_cv = image

        if isinstance(img_cv, Image.Image):
            img_cv = np.array(img_cv)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        if img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

        img_height, img_width = img_cv.shape[:2]

        if auto:
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            scale_factor = 0.6
            ws = (
                min(int(img_width * 1.05), int(screen_w * scale_factor)),
                min(int(img_height * 1.05), int(screen_h * scale_factor)),
            )
            wp = ((screen_w - ws[0]) // 2, (screen_h - ws[1]) // 2)

        window_name = "Image Viewer"
        constant_ws = ws

        zoom_factor = min(ws[0] / img_width, ws[1] / img_height)

        param["zoom_factor"] = zoom_factor

        param["mouse_left_click"] = False
        display_image(img_cv, param)


####### Figure Image Convertor (fic) #######

### fig                   (Figure)     Matplotlib figure object
### pi (pad_inches)       (float)      Padding around the image in inches, default is 1
### form (format)         (string)     Output image format (e.g., "png")

### return:
### img_cv                (ndarray)    Image array in OpenCV format


def fic(fig, form="png", pi=1):
    buf = BytesIO()
    fig.savefig(buf, format=form, bbox_inches="tight", pad_inches=pi * 0.3)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, 1)
    buf.close()
    return img_cv


####### Visualize Numerical Matrix (vnm) #######

### matrix                               (ndarray)    The matrix to be visualized (2D array, shape: n x m)
### cs (cell_size)                       (int)        Cell size scaling factor, default is 1
### f (font_size)                        (int)        Font size scaling factor, default is 1
### t (title_size)                       (int)        Title size scaling factor, default is 1
### k (tick_size)                        (int)        Axis tick font size scaling factor, default is 1
### cbs (color bar font size)            (int)        Color bar tick label size scaling factor, default is 1
### title                                (str)        Matrix title, default is "Numerical-Matrix"
### q (quality)                          (int)        Image DPI (quality), default is 300
### p (title_pad)                        (int)        Padding between title and matrix
### cmap (colormap)                      (str)        Colormap to use, default is "viridis"
### path (save_path)                     (str)        Optional save path, default is None
### sd (save_desktop)                    (bool)        Whether to save to desktop
### latex (use_latex)                    (bool)        Whether to use LaTeX rendering
### sxt (spacing of x-axis ticks)        (int)        Tick spacing on the x-axis, default is 1
### syt (spacing of y-axis ticks)        (int)        Tick spacing on the y-axis, default is 1


def vnm(
    matrix,
    cs=1,
    f=1,
    t=1,
    k=1,
    title="Numerical-Matrix",
    q=300,
    p=0,
    cmap="viridis",
    path=None,
    sd=0,
    latex=0,
    sxt=1,
    syt=1,
    cbs=1,
):

    rows, cols = matrix.shape

    fig_width = cols * cs * 0.1
    fig_height = rows * cs * 0.1
    title_fontsize = max(rows, cols) * 0.3 * t

    if latex == 1:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=q)

    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    font_size = cs * f * 19

    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]

            if np.iscomplexobj(matrix):
                real_part = value.real
                imag_part = value.imag

                if real_part == 0 and imag_part == 0:
                    display = "0"
                elif imag_part == 0:
                    display = f"{real_part:.2f}"
                elif real_part == 0:
                    display = f"{imag_part:.2f}i"
                elif imag_part > 0:
                    display = f"{real_part:.2f}+{imag_part:.2f}i"
                else:
                    display = f"{real_part:.2f}{imag_part:.2f}i"
            else:
                display = f"{value:.2f}" if value != 0 else "0"

            ax.text(
                j,
                i,
                display,
                ha="center",
                va="center",
                color="black" if display != "0" else "white",
                fontsize=font_size,
            )
    tick_fontsize = font_size * 6 * k
    ax.set_xticks(np.arange(0, cols, step=sxt))
    ax.set_yticks(np.arange(0, rows, step=syt))
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    cbar = fig.colorbar(cax)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    cbar.ax.tick_params(labelsize=cbs * tick_fontsize)

    ax.set_title(title, fontdict={"fontsize": title_fontsize}, pad=p)

    img_cv = fic(fig)
    vp([img_cv])

    try:
        if path and sd == 1:
            plt.savefig(path, bbox_inches="tight")
            print(f"Image saved to specified path: {path} (desktop flag overridden)")
        elif path:
            plt.savefig(path, bbox_inches="tight")
            print(f"Image saved to specified path: {path}")
        elif sd == 1:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            base_name = f"Matrix-{rows}x{cols}"
            init = "-0"
            ext = ".png"
            path = os.path.join(desktop_path, base_name + init + ext)

            # Check if file exists, append -0, -1, etc. if needed
            counter = 1
            while os.path.exists(path):
                path = os.path.join(desktop_path, f"{base_name}-{counter}{ext}")
                counter += 1

            plt.savefig(path, bbox_inches="tight")
            print(f"Image saved to desktop: {path}")
        else:
            print("No save path provided. Image was not saved.")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

    plt.close(fig)


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

        elif sd == 1:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            base_name = f"Symbol_Matrix-{rows}x{cols}"
            init = "-0"
            ext = ".png"
            path = os.path.join(desktop_path, base_name + init + ext)

            # Check if file exists, append -0, -1, etc. if needed
            counter = 1
            while os.path.exists(path):
                path = os.path.join(desktop_path, f"{base_name}-{counter}{ext}")
                counter += 1

            plt.savefig(path, bbox_inches="tight")
            print(f"Image saved to desktop: {path}")

        else:
            print("没有提供保存路径，图像未保存。")
    except Exception as e:
        print(f"保存图像时发生错误: {e}")

    plt.close(fig)  # 关闭图像，避免占用内存


####### LaTeX to txt #######

### latex              (str)       LaTeX 格式的文本, 需要转换为txt格式

import re
from pylatexenc.latex2text import LatexNodes2Text


def latexTtxt(latex):

    # 使用 pylatexenc 解析 LaTeX 文本
    sanitized = LatexNodes2Text().latex_to_text(latex)

    # 进一步清理文件系统非法字符
    sanitized = re.sub(r'[\\^&%$#@!*:<>?|"/]', "", sanitized)  # 移除文件系统非法字符
    sanitized = sanitized.replace(" ", "_")  # 替换空格为下划线

    return sanitized


####### Plot (LaTeX + matplotlib) #######

### x_data                   (ndarray)   x-axis data for plotting
### y_data                   (ndarray)   y-axis data for plotting
### title                    (str)       Title of the figure, supports LaTeX
### xlabel                   (str)       Label for the x-axis, supports LaTeX
### ylabel                   (str)       Label for the y-axis, supports LaTeX
### ts (title size)          (int)       Font size for the title
### tp (title pad)           (float)     Vertical distance between title and plot (pt)
### xs (xlabel size)         (int)       Font size for the x-axis label
### ys (ylabel size)         (int)       Font size for the y-axis label
### z  (zoom factor)         (float)     Scale factor for display size, default is 0.3
### s  (figure size)         (tuple)     Size of the figure (width, height), default is (7, 5)
### q  (quality)             (int)       Image resolution in DPI, default is 300
### plot_args                (dict/list) Additional plotting parameters passed to `plt.plot` or `plt.scatter`
### log_x                    (int/bool)  Set x-axis to logarithmic scale if True
### log_y                    (int/bool)  Set y-axis to logarithmic scale if True
### scatter                  (bool)      Scatter plot mode
### multi                    (bool/int)  Whether plotting multiple datasets, default is False
### show                     (bool)      Whether to open the image in a popup window
### ws (window size)         (tuple)     Viewer window size when using popup, default is (800, 500)
### wp (window position)     (tuple)     Viewer window position on screen, default is (600, 100)
### path (save_path)         (str)       Optional path to save the output file
### sd (save_desktop)        (bool)      If True, saves image to desktop if no path is specified


def plot(
    x_data,
    y_data,
    xlabel,
    ylabel,
    title,
    z=0.3,
    s=(7, 5),
    q=300,
    ts=16,
    tp=10,
    xs=11,
    ys=11,
    multi=0,
    plot_args=None,
    log_x=0,
    log_y=0,
    scatter=0,
    show=0,
    ws=(800, 500),
    wp=(600, 100),
    path=None,
    sd=0,
):

    original_backend = mpl.get_backend()

    mpl.use("pgf")

    pgf_with_latex = {
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "pgf.preamble": r"\usepackage{xeCJK}\setCJKmainfont{KaiTi} \usepackage{amsmath}",
    }

    mpl.rcParams.update(pgf_with_latex)

    if not multi:

        plt.figure(figsize=s)
        if plot_args is None:
            plot_args = {}
        if scatter:
            plt.scatter(x_data, y_data, **plot_args)
        else:
            plt.plot(x_data, y_data, **plot_args)
    else:
        if not isinstance(x_data, list) or not isinstance(y_data, list):
            raise ValueError(
                "In multi mode, x_data and y_data must be lists of arrays."
            )
        if plot_args is None:
            plot_args = [{}] * len(x_data)
        for x, y, args in zip(x_data, y_data, plot_args):
            if scatter:
                plt.scatter(x, y, **args)
            else:
                plt.plot(x, y, **args)

    if log_y == 1:
        plt.yscale("log")

    if log_x == 1:
        plt.xscale("log")

    plt.xlabel(xlabel, fontsize=xs)
    plt.ylabel(ylabel, fontsize=ys)
    plt.title(title, fontsize=ts, pad=tp)

    plt.grid(True)

    if multi == 1:
        has_label = any("label" in str(args).lower() for args in plot_args)
    else:
        has_label = "label" in str(plot_args).lower()

    if has_label:
        plt.legend()

    (
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        if not log_y
        else None
    )
    (
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        if not log_x
        else None
    )

    plt.tight_layout()

    img_buffer = io.BytesIO()

    plt.savefig(img_buffer, format="png", dpi=q)

    plt.close("all")

    img_buffer.seek(0)

    image = Image.open(img_buffer)
    original_width, original_height = image.size

    new_width = int(original_width * z)
    new_height = int(original_height * z)

    display(IPImage(data=img_buffer.getvalue(), width=new_width, height=new_height))

    if show == 1:
        vp(image, ws, wp)

    try:
        if path and sd == 1:
            with open(path, "wb") as f:
                f.write(img_buffer.getvalue())
            print(
                f"Image saved to specified path: {path} (custom path was used even though sd == 1)."
            )
        elif path:
            with open(path, "wb") as f:
                f.write(img_buffer.getvalue())
            print(f"Image saved to specified path: {path}")
        elif sd == 1:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            path = os.path.join(desktop_path, f"{latexTtxt(title)}.png")
            with open(path, "wb") as f:
                f.write(img_buffer.getvalue())
            print(f"Image saved to desktop: {path}")
        else:
            print("No save path provided. Image was not saved.")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

    img_buffer.close()
    mpl.use(original_backend)


####### Horn #######

### text                       (str)
### dpi                        (int)
### fz(fontsize)               (int)
### w(window_width)            (int)
### h(window_height)           (int)
### t(display_duration/ms)     (int)


def Horn(text, fz=12, dpi=1000, w=800, h=300, t=1370):
    # Adjust figure size to fit the text size (narrow rectangle)
    fig = plt.figure(figsize=(4.53, 1.3137), facecolor="black", dpi=dpi)

    # Create a plot and add the text in white with Times New Roman italic
    plt.text(
        0.5,
        0.4,
        text,
        fontsize=fz,
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


####### To Quick Response Code (tqr) #######

### data               (str)                  The input string or URL to encode as a QR code


def tqr(data: str):
    # Generate QR code
    img_qr = qrcode.make(data)

    # Convert PIL image to OpenCV BGR image
    img_cv = cv2.cvtColor(np.array(img_qr.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Display the image
    vp(img_cv)


####### Portal_document_Format Resampleing (pdf_rs) #######

##### Dependency

# qpdf_path = r"C:\2SilverHand\qpdf\bin\qpdf.exe"
# gs_path = r"C:\2SilverHand\ghostpdl\gs10.05.1\bin\gswin64c.exe"

### pdf_paths          (list[str | Path])     PDFs to process; order is preserved in the return.

### wts                (int)                  Save mode:
#                                              - 0 : Save alongside source (auto `_N` suffix). If `output_dir`
#                                                   is given, save there instead (still uses `_N` to avoid collisions).
#                                              - 1 : Replace original atomically (write temp, then move over on success).

### output_dir         (str | Path | None)    Target directory when `wts == 0`. None -> next to source file(s).

### qpdf_opts          (dict | None)          QPDF options override. Keys = numeric shorthands or full names.
#                                               Values:
#                                               • None   -> include as a flag (no value)
#                                               • str/int-> include as `key=value`
#                                               • False  -> drop/disable this option

### gs_opts            (dict | None)          Ghostscript options override. Same rules as `qpdf_opts`.
#                                             Common: {"-dPDFSETTINGS": "/screen|/ebook|/printer|/prepress"}.

### keep_temp          (bool)                 Keep intermediate working directory (QPDF outputs, temp files).
#                                             If True, the function will print/log the temp dir path.

### qpdf_path          (str | Path)           Path to `qpdf` executable.
### gs_path            (str | Path)           Path to Ghostscript CLI (e.g., `gswin64c.exe` on Windows).
### Returns            (list[Path])           Output PDF paths, same order as `pdf_paths`.


def pdf_rs(
    pdf_paths,
    wts=0,
    output_dir=None,
    qpdf_opts=None,  # dict: numeric keys (preferred) or full option names
    gs_opts=None,  # dict: numeric keys (preferred) or full option names
    keep_temp=False,
    qpdf_path=r"C:\2SilverHand\qpdf\bin\qpdf.exe",
    gs_path=r"C:\2SilverHand\ghostpdl\gs10.05.1\bin\gswin64c.exe",
):

    # ---------- Shorthand maps (code -> real option name) ----------
    qpdf_map = {
        1: "--warning-exit-0",
        2: "--stream-data",
        3: "--object-streams",
        4: "--optimize-images",
        5: "--jpeg-quality",
    }
    gs_map = {
        1: "-sDEVICE",
        2: "-dCompatibilityLevel",
        3: "-dPDFSETTINGS",  # /screen /ebook /printer /prepress
        4: "-dNOPAUSE",
        5: "-dQUIET",
        6: "-dBATCH",
        # 7: "-sOutputFile"  # handled separately at runtime
    }

    # ---------- Default options using shorthand codes ----------
    qpdf_defaults = {
        1: None,  # --warning-exit-0
        2: "compress",  # --stream-data=compress
        3: "generate",  # --object-streams=generate
        4: None,  # --optimize-images
        5: "5",  # --jpeg-quality=5
    }
    gs_defaults = {
        1: "pdfwrite",  # -sDEVICE=pdfwrite
        2: "1.3",  # -dCompatibilityLevel=1.3
        3: "/prepress",  # -dPDFSETTINGS=/screen|/ebook|/printer|/prepress
        4: None,  # -dNOPAUSE
        5: None,  # -dQUIET
        6: None,  # -dBATCH
    }

    # ---------- Helpers ----------
    def _as_path(p):
        return Path(p) if not isinstance(p, Path) else p

    def _run(cmd, title):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.stdout.strip():
            print(f"[{title}] STDOUT:\n{res.stdout.strip()}")
        if res.stderr.strip():
            print(f"[{title}] STDERR:\n{res.stderr.strip()}")
        if res.returncode != 0:
            raise RuntimeError(f"{title} failed with code {res.returncode}")

    def _increment_suffix(path: Path) -> Path:
        # name.pdf -> name_1.pdf; name_2.pdf -> name_3.pdf
        m = re.search(r"(.*)_(\d+)$", path.stem)
        if m:
            base, n = m.group(1), int(m.group(2)) + 1
            return path.with_name(f"{base}_{n}{path.suffix}")
        else:
            return path.with_name(f"{path.stem}_1{path.suffix}")

    def _target_path(src: Path) -> Path:
        if wts == 1:
            return src
        base = (Path(output_dir) if output_dir else src.parent) / src.name
        out = _increment_suffix(base)
        while out.exists():
            out = _increment_suffix(out)
        return out

    def _normalize_overrides(overrides, opt_map):
        """
        Accept numeric or full-name keys, return a dict keyed by real names.
        Value False removes option. None is flag. Else becomes string.
        """
        if not overrides:
            return {}
        norm = {}
        for k, v in overrides.items():
            if isinstance(k, int):
                if k not in opt_map:
                    raise KeyError(f"Unknown shorthand key: {k}")
                name = opt_map[k]
            else:
                name = str(k)
            norm[name] = v
        return norm

    def _merge_opts(defaults, opt_map, overrides):
        """
        Convert shorthand defaults to a dict keyed by real names,
        then apply normalized overrides (which may delete with False).
        """
        merged = {opt_map[k]: v for k, v in defaults.items()}
        for name, val in overrides.items():
            if val is False:
                merged.pop(name, None)  # remove option
            else:
                # Normalize some values, e.g., PDFSETTINGS allow "ebook" -> "/ebook"
                if (
                    name == "-dPDFSETTINGS"
                    and isinstance(val, str)
                    and not val.startswith("/")
                ):
                    val = "/" + val
                merged[name] = val
        return merged

    def _build_cmd(exe, merged_opts, trailing_args=None):
        """
        Build command list: flags (None) -> just the name; valued -> name=value.
        Order follows gs_map/qpdf_map numerical order, then any extras.
        """
        cmd = [str(exe)]
        # Respect preferred order defined by the map; then append others deterministically.
        order = []
        # choose the correct map by probing a key
        sample = next(iter(merged_opts.keys()), None)
        current_map = (
            gs_map
            if sample and sample.startswith("-d") or sample == "-sDEVICE"
            else qpdf_map
        )

        for code in current_map:
            name = current_map[code]
            if name in merged_opts:
                val = merged_opts[name]
                if val is None:
                    cmd.append(name)
                else:
                    cmd.append(f"{name}={val}")
                order.append(name)
        # Append any options not present in the map (or extra custom ones)
        for name, val in merged_opts.items():
            if name in order:
                continue
            if val is None:
                cmd.append(name)
            else:
                cmd.append(f"{name}={val}")

        if trailing_args:
            cmd.extend(map(str, trailing_args))
        return cmd

    # ---------- Validate inputs ----------
    if not isinstance(pdf_paths, (list, tuple)) or not pdf_paths:
        raise ValueError("pdf_paths must be a non-empty list")
    pdf_paths = [_as_path(p) for p in pdf_paths]
    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {p}")

    qpdf_exec = _as_path(qpdf_path)
    gs_exec = _as_path(gs_path)
    if not qpdf_exec.exists():
        raise FileNotFoundError(f"QPDF not found: {qpdf_exec}")
    if not gs_exec.exists():
        raise FileNotFoundError(f"Ghostscript not found: {gs_exec}")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Merge options ----------
    qpdf_over = _normalize_overrides(qpdf_opts, qpdf_map)
    gs_over = _normalize_overrides(gs_opts, gs_map)

    qpdf_merged = _merge_opts(qpdf_defaults, qpdf_map, qpdf_over)
    gs_merged = _merge_opts(gs_defaults, gs_map, gs_over)

    results = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_rs_"))

    try:
        for src in pdf_paths:
            qpdf_out = tmp_dir / f"{src.stem}_qpdf.pdf"
            target = _target_path(src)

            # -------- QPDF stage --------
            qpdf_cmd = _build_cmd(
                qpdf_exec, qpdf_merged, trailing_args=[str(src), str(qpdf_out)]
            )
            _run(qpdf_cmd, f"QPDF: {src.name}")

            # -------- Ghostscript stage --------
            if wts == 1:
                tmp_final = src.with_name(src.stem + ".__tmp__.pdf")
                out_arg = "-sOutputFile=" + str(tmp_final)
            else:
                out_arg = "-sOutputFile=" + str(target)

            # Build GS command: options first, then add output and input
            gs_cmd = _build_cmd(
                gs_exec, gs_merged, trailing_args=[out_arg, str(qpdf_out)]
            )
            _run(gs_cmd, f"Ghostscript: {src.name}")

            if wts == 1:
                tmp_final.replace(src)
                target = src

            if not keep_temp:
                qpdf_out.unlink(missing_ok=True)

            results.append(target.resolve())

        return results

    finally:
        if not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
