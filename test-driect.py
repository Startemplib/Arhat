from arpy.visual import *

random_matrix = np.random.rand(10, 30)  # 生成100x100的随机矩阵
vnm(random_matrix, title="仿宋", q=500, sd=1)


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
# vp(image_objects)
