# ---------------------- 文件操作和路径处理 ----------------------
import os  # 用于文件和目录的操作

# ---------------------- 图像处理和绘图工具 ----------------------
import cv2  # OpenCV用于图像处理
import numpy as np  # 用于矩阵和数值操作
import matplotlib.pyplot as plt  # 用于绘制图表
from matplotlib.font_manager import FontProperties  # 用于自定义字体属性
from matplotlib.ticker import FuncFormatter  # 用于自定义刻度格式
import matplotlib.ticker as ticker  # 用于设置图表刻度

# ---------------------- 内存操作和格式转换 ----------------------
from io import BytesIO  # 用于在内存中处理二进制数据
import re  # 正则表达式库，用于字符串匹配和处理

# ---------------------- 符号计算工具 ----------------------
import sympy as sp  # SymPy用于符号矩阵和代数运算


# ---------------------- Arhat ----------------------

# Adjust figure size to fit the text size (narrow rectangle)
fig = plt.figure(figsize=(4.53, 1.3137), facecolor="black")

# Create a plot and add the text in white with Times New Roman italic
plt.text(
    0.5,
    0.4,
    """"Breaking through the Empyrean."
    
    —— Arhat is here
    """,
    fontsize=12,
    fontfamily="Times New Roman",
    fontstyle="italic",
    color="white",
    ha="center",
    va="center",
)

# Hide the axes
plt.gca().set_axis_off()

# Show the plot with the styled text
plt.show()


__all__ = ["visual", "matrix"]
