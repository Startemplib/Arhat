from arpy.visual import *

# 使用 arpy 模块中的 vnm 函数
matrix = np.array([[1, 2], [3, 4]])  # 将列表转换为 NumPy 数组
vnm(matrix, "Test Matrix")


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
vp(image_objects)
