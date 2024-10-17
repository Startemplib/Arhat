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
from arpy.visual import *

horn(
    """"Breaking through the Empyrean."
    
    —— Arhat is here
    """,
    w=800,
    h=300,
    t=1370,
)

__all__ = ["visual", "matrix"]
