import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 在 -0.5 到 0.5 之间等间距取50个点作为x值（保持不变）
x = np.linspace(-0.5, 0.5, 50)

# 调整函数参数，使y自然落在[-0.5, 0.5]范围内（无需缩放）
# 公式：y = A*cos(4πx) + Bx，通过A=0.25、B=0.5控制范围
y = 0.25 * np.cos(4 * np.pi * x) + 0.5 * x  # 核心修改：调整系数A和B

# 绘制散点图（使用原始y值，无缩放）
plt.scatter(x, y, color='orange')

# 保持原标题和字体设置不变
plt.title('Cos(x) + x', fontsize=20)
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示

# 保持原坐标轴范围和刻度设置不变
plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

# 不显示网格线（保持原设置）
plt.grid(False)
plt.tight_layout()  # 优化布局，防止字体溢出
plt.show()

# 创建DataFrame并保存（y为原始函数值，可通过公式复现）
data = {'x': x, 'y': y}
df = pd.DataFrame(data)
df.to_excel('Cosx-True.xlsx', index=False)  # 保持原文件名
