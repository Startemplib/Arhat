import numpy as np  # 用于矩阵操作
import matplotlib.pyplot as plt  # 用于绘图


def vnm(
    matrix,
    cell_size=0.3,
    r=17,
    title="Matrix Viewer",
    title_fontsize=20,
    dpi=300,
    cmap="viridis",
):
    rows, cols = matrix.shape
    # 每个格子的基准尺寸
    fig_width = cols * cell_size
    fig_height = rows * cell_size

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

    # 隐藏默认的主轴刻度
    ax.tick_params(which="minor", size=0)

    fig.colorbar(cax)
    ax.set_title(title, fontsize=title_fontsize)
    plt.show()


import os
import cv2
import numpy as np


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
