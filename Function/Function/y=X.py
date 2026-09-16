import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 在 -0.5 到 0.5 之间等间距取21个点作为x值
x = np.linspace(-0.5, 0.5, 21)
# 计算y值，根据y = |x|
y = np.abs(x)

# 绘制散点图
plt.scatter(x, y, color='orange')

# 调整字体大小参数
plt.title('|x|', fontsize=20)  # 标题字体大小设为20
plt.xlabel('', fontsize=20)    # 取消x轴字母显示
plt.ylabel('', fontsize=20)    # 取消y轴字母显示

# 设置x轴刻度范围和刻度标签（刻度字体大小设为28）
plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)

# 设置y轴刻度范围和刻度标签（刻度字体大小设为28）
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

# 不显示网格线
plt.grid(False)
plt.tight_layout()  # 优化布局
plt.show()

# 创建DataFrame
data = {'x': x, 'y': y}
df = pd.DataFrame(data)

# 保存到Excel文件
df.to_excel('abs_x.xlsx', index=False)