import numpy as np  # 用于矩阵操作
import matplotlib.pyplot as plt  # 用于绘图
import os
import cv2
from io import BytesIO
import matplotlib.ticker as ticker


def get_desktop_path():
    """获取当前用户桌面的路径"""
    return os.path.join(os.path.expanduser("~"), "Desktop")


def vp(images, window_size=(1000, 800), window_position=(1000, 100)):
    # 全局变量
    window_name = "Image Viewer"
    constant_window_size = window_size  # 窗口大小可由参数输入

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
        relative_mouse_x = mouse_x / constant_window_size[0]
        relative_mouse_y = mouse_y / constant_window_size[1]
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        background = (
            np.ones(
                (constant_window_size[1], constant_window_size[0], 3), dtype=np.uint8
            )
            * 255
        )
        resized = cv2.resize(img_cv, (new_w, new_h))

        start_x = max(0, int(relative_mouse_x * new_w - constant_window_size[0] // 2))
        start_y = max(0, int(relative_mouse_y * new_h - constant_window_size[1] // 2))
        end_x = min(start_x + constant_window_size[0], new_w)
        end_y = min(start_y + constant_window_size[1], new_h)

        cropped_resized = resized[start_y:end_y, start_x:end_x]
        background[0 : cropped_resized.shape[0], 0 : cropped_resized.shape[1]] = (
            cropped_resized
        )

        return background

    # 显示图像的函数
    def display_image(img_cv, param):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, *window_position)  # 使用传入的窗口位置
        cv2.resizeWindow(window_name, *constant_window_size)

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


def fti(fig, pad_inches=0.1):
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


def vnm(
    matrix,
    cell_size=0.3,
    r=17,
    t=1,
    k=1,
    title="Matrix Viewer",
    dpi=300,
    cmap="viridis",
    save_path=None,  # 用户可以指定保存路径
    save_to_desktop=0,  # 增加是否保存到桌面的选项，默认为0不保存
):
    rows, cols = matrix.shape
    # 每个格子的基准尺寸
    fig_width = cols * cell_size
    fig_height = rows * cell_size
    title_fontsize = max(rows, cols) * 0.3 * t

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    # 自动调整字体大小
    font_size = cell_size * r  # 动态调整字体大小
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
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    cbar.ax.tick_params(labelsize=tick_fontsize)
    ax.set_title(title, fontdict={"fontsize": title_fontsize})

    img_cv = fti(fig)  # 将 figure 转换为 OpenCV 格式的图像
    vp([img_cv])  # 使用 vp 函数显示图像

    if save_to_desktop == 1:  # 检查是否要保存到桌面
        desktop_path = get_desktop_path()
        save_path = os.path.join(desktop_path, f"Matrix-{rows}x{cols}.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"图像已保存到桌面: {save_path}")
    elif save_path:  # 如果提供了保存路径
        plt.savefig(save_path, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")

    plt.close(fig)  # 关闭图像，避免占用内存
