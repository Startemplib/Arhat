from arpy import *

# 示例数据
matrix = np.random.rand(15, 15)  # 5x5 随机矩阵

# 示例1：默认渲染方式（不启用 LaTeX）
vnm(matrix, title="Matrix Viewer", latex=0)

# 示例2：启用 LaTeX 渲染方式
latex_title = r"$\textbf{Matrix}\ \alpha + \beta = \gamma$"  # LaTeX 格式的标题
vnm(matrix, title=latex_title, latex=1)
