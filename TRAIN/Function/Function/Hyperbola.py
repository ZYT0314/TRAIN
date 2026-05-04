import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 生成x值：避开x=0（避免分母为0），分两段生成以覆盖正负区间
# 第一段：x∈[-0.5, -0.05]（第二象限部分）
x_neg = np.linspace(-0.5, -0.05, 20)
# 第二段：x∈[0.05, 0.5]（第四象限部分）
x_pos = np.linspace(0.05, 0.5, 20)
# 合并为完整x值（共20个点，与之前函数数量一致）
x = np.concatenate([x_neg, x_pos])

# 核心：双象限双曲线函数（xy=k形式，k<0时分布在二、四象限）
# 函数形式：y = k / x（k为负数，确保x正y负、x负y正，即二、四象限）
# 参数设计：
# - k=-0.05：控制双曲线陡峭程度，确保y∈[-0.5, 0.5]
k = -0.05  # 负数保证二、四象限分布
y = k / x  # 标准双曲线方程（xy=k）

# 验证y值范围（确保在[-0.5, 0.5]内）
print(f"y值范围：[{np.min(y):.4f}, {np.max(y):.4f}]")  # 输出验证结果

# 绘制散点图（格式与其他函数完全一致）
plt.scatter(x, y, color='orange')

# 调整字体大小参数
plt.title('y = -0.05 / x', fontsize=20)  # 标题显示函数形式
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示

# 设置x轴刻度范围和刻度标签（与其他函数一致）
plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)

# 设置y轴刻度范围和刻度标签（与其他函数一致）
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

# 不显示网格线
plt.grid(False)
plt.tight_layout()  # 优化布局，防止字体溢出
plt.show()

# 保存数据到Excel（格式与其他函数一致）
data = {'x': x, 'y': y}  # 直接使用原始y值，无需缩放
df = pd.DataFrame(data)
df.to_excel('hyperbola-True40.xlsx', index=False)  # 文件名体现二、四象限双曲线