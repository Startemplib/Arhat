from arpy.visual import *

# random_matrix = np.random.rand(10, 30)  # 生成100x100的随机矩阵
# vnm(random_matrix, title="仿宋", q=500, sd=1)


# 调用函数时可指定窗口大小和位置
# 你可以输入图像路径列表，或者已经加载的图像
image_paths = [
    r"C:\Users\lj\Desktop\image1.png",
    r"C:\Users\lj\Desktop\image2.png",
]
# 或者输入已经加载的图像，例如：
# image_objects = [
#    cv2.imread(r"C:\Users\lj\Desktop\image1.png"),
#    cv2.imread(r"C:\Users\lj\Desktop\image2.png"),
# ]

# 显示图片
# vp(image_paths, ws=(1200, 900), wp=(500, 200))
# vp(image_objects)

# 示例数据
matrix = np.random.rand(100, 100)  # 5x5 随机矩阵

# 示例1：默认渲染方式（不启用 LaTeX）
# vnm(matrix, title="Matrix Viewer", latex=0)

# 示例2：启用 LaTeX 渲染方式
latex_title = r"$\textbf{Matrix}\ \alpha + \beta = \gamma$"  # LaTeX 格式的标题
vnm(matrix, title=latex_title, latex=1)
