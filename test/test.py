import matplotlib.pyplot as plt

# 启用 LaTeX 数学模式
plt.rcParams["text.usetex"] = True

# 创建一个简单的图
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)

# 设置带有 LaTeX 的标题和标签
plt.title(r"$y = x^2$", fontsize=14)  # LaTeX 数学公式
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)

plt.show()
