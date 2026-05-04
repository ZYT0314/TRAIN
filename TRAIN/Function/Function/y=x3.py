import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 在 -0.5 到 0.5 之间等间距取20个点作为x值
x = np.linspace(-0.5, 0.5, 20)

# 三次函数：y = 4x³
# 参数说明：当x∈[-0.5,0.5]时，x³∈[-0.125,0.125]，4x³∈[-0.5,0.5]，自然满足范围要求
y = 4 * x**3  # 核心：通过系数4控制范围，无需缩放

# 绘制散点图
plt.scatter(x, y, color='orange')

# 调整字体大小参数
plt.title('4x³', fontsize=20)  # 标题更新为三次函数
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示

# 设置x轴刻度范围和刻度标签
plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)

# 设置y轴刻度范围和刻度标签
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

# 不显示网格线
plt.grid(False)
plt.tight_layout()  # 优化布局，防止字体溢出
plt.show()

# 创建DataFrame
data = {'x': x, 'y': y}  # 直接使用原始y值，无需缩放
df = pd.DataFrame(data)

# 保存到Excel文件
df.to_excel('ax3.xlsx', index=False)
    