import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义区间边界
x_boundaries = np.linspace(-0.5, 0.5, 6)
# 每个阶梯对应4个数据点，总共20个点
points_per_step = 4
total_points = len(x_boundaries) * points_per_step

# 生成均匀分布的x值
x = np.zeros(total_points)
y = np.zeros(total_points)
for i in range(len(x_boundaries) - 1):
    start_idx = i * points_per_step
    end_idx = (i + 1) * points_per_step
    x[start_idx:end_idx] = np.linspace(x_boundaries[i], x_boundaries[i + 1], points_per_step)
    if i < 5:
        y[start_idx:end_idx] = [(-0.4 + i * 0.2)] * points_per_step

# 处理数据以绘制线段
x_segments = []
y_segments = []
start_index = 0
for i in range(1, len(x)):
    if y[i] != y[i - 1]:
        x_segments.append(x[start_index:i])
        y_segments.append([y[start_index]] * (i - start_index))
        start_index = i
x_segments.append(x[start_index:])
y_segments.append([y[start_index]] * (len(x) - start_index))

# 绘制线段，设置线宽为2
for x_seg, y_seg in zip(x_segments, y_segments):
    plt.plot(x_seg, y_seg, color='orange', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Step function with five gradients')

# 设置x轴刻度范围和刻度标签
plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5])

# 设置y轴刻度范围和刻度标签
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5])

# 不显示网格线
plt.grid(False)

plt.show()

# 创建DataFrame
data = {'x': x, 'y': y}
df = pd.DataFrame(data)

# 保存到Excel文件
df.to_excel('step_function_five_gradients.xlsx', index=False)