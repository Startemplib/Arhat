import os
import cv2
import numpy as np

# 全局变量
zoom_factor = 1.0
mouse_x, mouse_y = 0, 0
mouse_left_click = False
window_name = "Image Viewer"
constant_window_size = (1000, 800)  # 窗口恒定大小 (宽, 高)


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    global zoom_factor, mouse_x, mouse_y, mouse_left_click
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_factor += 0.1  # 增加缩放因子
        else:
            zoom_factor = max(0.1, zoom_factor - 0.1)  # 限制最小缩放因子
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_left_click = True


# 调整图像以保持鼠标为中心的缩放
def adjust_to_mouse_center(img_cv, zoom_factor):
    global mouse_x, mouse_y
    h, w, _ = img_cv.shape
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)

    # 创建白色背景
    background = (
        np.ones((constant_window_size[1], constant_window_size[0], 3), dtype=np.uint8)
        * 255
    )

    # 调整后的图像
    resized = cv2.resize(img_cv, (new_w, new_h))

    # 计算放置位置，确保图像居中
    offset_x = max(0, (constant_window_size[0] - new_w) // 2)
    offset_y = max(0, (constant_window_size[1] - new_h) // 2)

    # 确保不超出背景边界，并进行裁剪
    end_x = offset_x + new_w
    end_y = offset_y + new_h

    if end_x > constant_window_size[0]:
        end_x = constant_window_size[0]
    if end_y > constant_window_size[1]:
        end_y = constant_window_size[1]

    resized_cropped = resized[
        0 : end_y - offset_y, 0 : end_x - offset_x
    ]  # 裁剪调整后的图像

    # 将调整后的图像放置到背景上
    background[offset_y:end_y, offset_x:end_x] = resized_cropped

    return background


# 显示图像的函数
def display_image(image_path):
    global zoom_factor, mouse_left_click
    mouse_left_click = False

    img = cv2.imread(image_path)
    if img is None:
        print(f"加载图像错误: {image_path}")
        return

    # 设置窗口大小
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1000, 100)  # 设置窗口初始位置
    cv2.resizeWindow(window_name, *constant_window_size)  # 初始窗口大小

    while True:
        adjusted_img = adjust_to_mouse_center(img, zoom_factor)
        cv2.imshow(window_name, adjusted_img)

        # 根据调整后的图像尺寸调整窗口大小
        h, w, _ = adjusted_img.shape
        cv2.resizeWindow(window_name, w, h)  # 使窗口大小适应图像

        cv2.setMouseCallback(window_name, mouse_callback)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or mouse_left_click:  # 按下 Enter 键或左键单击关闭
            cv2.destroyAllWindows()
            break


# 获取图片文件
def get_image_files(directory):
    return [
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


# 主逻辑
directory = r"C:\Users\lj\Desktop"  # 替换为你的图片目录
image_files = get_image_files(directory)

# 展示每张图片
for image_file in image_files:
    display_image(os.path.join(directory, image_file))
