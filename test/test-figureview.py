import os
import cv2
import numpy as np


def display_images(images, window_size=(1000, 800), window_position=(1000, 100)):
    # 全局变量
    window_name = "Image Viewer"
    constant_window_size = window_size  # 窗口大小可由参数输入

    # 定义鼠标事件回调函数
    def mouse_callback(event, x, y, flags, param):
        param["mouse_x"], param["mouse_y"] = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                param["zoom_factor"] += 0.1
            else:
                param["zoom_factor"] = max(0.1, param["zoom_factor"] - 0.1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            param["mouse_left_click"] = True

    # 调整图像以保持鼠标为中心的缩放，允许 x, y 小于 0，填充空白
    def adjust_to_mouse_center(img_cv, zoom_factor, mouse_x, mouse_y):
        h, w, _ = img_cv.shape
        relative_mouse_x = mouse_x / constant_window_size[0]
        relative_mouse_y = mouse_y / constant_window_size[1]
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        # 创建白色背景
        background = (
            np.ones(
                (constant_window_size[1], constant_window_size[0], 3), dtype=np.uint8
            )
            * 255
        )

        # 缩放图像
        resized = cv2.resize(img_cv, (new_w, new_h))

        # 计算图像在背景上的起始点，允许 x 和 y 小于 0
        start_x = int(relative_mouse_x * new_w - constant_window_size[0] // 2)
        start_y = int(relative_mouse_y * new_h - constant_window_size[1] // 2)

        # 确定裁剪区域，确保不超出图像范围
        crop_x1 = max(0, -start_x)
        crop_y1 = max(0, -start_y)
        crop_x2 = min(new_w, constant_window_size[0] - start_x)
        crop_y2 = min(new_h, constant_window_size[1] - start_y)

        # 确定背景放置图像的起始点，确保不超出背景范围
        bg_x1 = max(0, start_x)
        bg_y1 = max(0, start_y)

        # 确保裁剪后的区域尺寸不为零
        cropped_width = crop_x2 - crop_x1
        cropped_height = crop_y2 - crop_y1

        if cropped_width > 0 and cropped_height > 0:
            # 将裁剪后的图像部分放到背景上
            background[
                bg_y1 : bg_y1 + cropped_height, bg_x1 : bg_x1 + cropped_width
            ] = resized[crop_y1:crop_y2, crop_x1:crop_x2]

        return background

    # 显示图像的函数
    def display_image(img_cv, param):
        if img_cv is None:
            print("图像加载失败，跳过显示。")
            return

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
            if key == 13 or param["mouse_left_click"]:  # 按下 Enter 或单击关闭
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


# 调用函数时可指定窗口大小和位置
# 你可以输入图像路径列表，或者已经加载的图像
image_paths = [
    r"C:\Users\lj\Desktop\image1.png",
    r"C:\Users\lj\Desktop\image2.png",
]
# 或者输入已经加载的图像，例如：
image_objects = [
    cv2.imread(r"C:\Users\lj\Desktop\image1.png"),
    cv2.imread(r"C:\Users\lj\Desktop\image2.png"),
]

# 显示图片
# display_images(image_paths, window_size=(1200, 900), window_position=(500, 200))
display_images(image_objects)
