import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义区间边界
x_boundaries = np.linspace(-0.5, 0.5, 6)
# 每个阶梯对应20个数据点
points_per_step = 20
total_points = len(x_boundaries) * points_per_step

# 生成均匀分布的x值
x = np.zeros(total_points)
y = np.zeros(total_points)
for i in range(len(x_boundaries) - 1):
    start_idx = i * points_per_step
    end_idx = (i + 1) * points_per_step
    # 计算每个小区间的步长
    step = (x_boundaries[i + 1] - x_boundaries[i]) / (points_per_step + 1)
    # 生成当前区间内均匀分布的x值
    x[start_idx:end_idx] = [x_boundaries[i] + (j + 1) * step for j in range(points_per_step)]
    # 计算当前区间对应的y值
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

# 设置图形参数
plt.title('Step function with five gradients', fontsize=20)
plt.xlabel('', fontsize=20)
plt.ylabel('', fontsize=20)
plt.grid(False)

plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

plt.tight_layout()
plt.show()

# 创建DataFrame并保存到Excel
data = {'x': x, 'y': y}
df = pd.DataFrame(data)
df.to_excel('step_function_five_gradients.xlsx', index=False)    