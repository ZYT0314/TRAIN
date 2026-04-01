import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np  # 用于刻度计算

# 读取数据
features_df = pd.read_excel('随访转录分析-V2.xlsx')
shap_values_df = pd.read_excel('feature_shap_随访转录分析-V2.xlsx')

# 确保特征名一致
assert features_df.columns.tolist() == shap_values_df.columns.tolist(), "特征名不一致"
feature_names = features_df.columns.tolist()
X = features_df.values
shap_values = shap_values_df.values

# ----------------------
# 关键参数设置 - 这里修改会生效
# ----------------------
plot_width = 12        # 图形宽度，越大越宽
plot_height = 8        # 图形高度，越大越高
font_size = 25         # 基础字体大小
label_font_size = 25   # 轴标签字体大小（x轴标题和右侧标题共用）
tick_font_size = 22    # 刻度字体大小
colorbar_label_size = 25  # 颜色柱标签字体大小
scatter_size = 100     # 散点大小，数值越大散点越大

# 设置matplotlib全局参数
mpl.rcParams['figure.figsize'] = (plot_width, plot_height)
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['lines.markersize'] = scatter_size / 10  # 全局标记大小设置

# 创建自定义字体对象
custom_font = FontProperties(size=font_size)
label_font = FontProperties(size=label_font_size)  # x轴标题和右侧标题共用
tick_font = FontProperties(size=tick_font_size)
colorbar_font = FontProperties(size=colorbar_label_size)

# 关闭所有之前的图形，确保全新开始
plt.close('all')
# 创建图形并设置大小
fig = plt.figure(figsize=(plot_width, plot_height))

# 绘制SHAP图 - 不使用任何散点大小参数（兼容最旧版本）
shap.summary_plot(
    shap_values, 
    X, 
    feature_names=feature_names,
    show=False,
    plot_size=(plot_width, plot_height)
)

# 获取主坐标轴
ax = plt.gca()

# 关键修改：直接找到散点对象并修改其大小
# 遍历所有图形元素，找到散点集合
for collection in ax.collections:
    # 检查是否为散点集合
    if hasattr(collection, 'set_sizes'):
        collection.set_sizes([scatter_size])  # 设置散点大小

# 调整y轴特征名称字体
for label in ax.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_size)

# ----------------------
# 核心优化：智能刻度分配（根据0点左右范围动态调整刻度数量）
# ----------------------
# 获取当前x轴的范围
x_min, x_max = ax.get_xlim()

# 确保0点在坐标轴范围内（避免0点超出原始范围）
new_x_min = min(x_min, 0)
new_x_max = max(x_max, 0)

# 计算0点左右两侧的实际范围（距离）
left_range = abs(new_x_min - 0)  # 0点到左侧最小值的距离
right_range = abs(new_x_max - 0)  # 0点到右侧最大值的距离

# 初始化刻度列表，确保0点始终存在
ticks = [0.0]

# 智能分配刻度：范围小的一侧1个刻度，范围大的一侧2个刻度（总4个）
if left_range <= right_range:
    # 左侧范围小 → 左侧1个刻度，右侧2个刻度
    # 左侧添加1个刻度（最左侧极值）
    if left_range > 0:
        ticks.append(new_x_min)
    # 右侧添加2个刻度（极值+中间值）
    if right_range > 0:
        ticks.append(new_x_max)  # 右侧极值
        ticks.append(new_x_max - right_range * 0.5)  # 右侧中间值（靠近0点）
else:
    # 右侧范围小 → 右侧1个刻度，左侧2个刻度
    # 右侧添加1个刻度（最右侧极值）
    if right_range > 0:
        ticks.append(new_x_max)
    # 左侧添加2个刻度（极值+中间值）
    if left_range > 0:
        ticks.append(new_x_min)  # 左侧极值
        ticks.append(new_x_min + left_range * 0.5)  # 左侧中间值（靠近0点）

# 去重+排序（避免重复刻度，确保顺序正确）
ticks = sorted(list(set(ticks)))

# 特殊情况处理：若因范围为0（如所有值=0）导致刻度不足4个，补充均匀刻度
while len(ticks) < 4:
    if len(ticks) >= 2:
        # 按已有刻度间距补充
        interval = ticks[-1] - ticks[-2]
        ticks.append(ticks[-1] + interval)
    else:
        # 极端情况（仅0点）：手动添加均匀刻度
        ticks.extend([-0.5, 0.5, 1.0])  # 临时值，后续会排序去重
    ticks = sorted(list(set(ticks)))

# 确保最终仅保留4个刻度
ticks = ticks[:4]

# 动态调整小数位数（根据总数据范围适配精度）
data_range = new_x_max - new_x_min
if data_range >= 10:
    decimal_places = 0  # 大范围：无小数
elif data_range >= 1:
    decimal_places = 1  # 中等范围：1位小数
elif data_range >= 0.1:
    decimal_places = 2  # 小范围：2位小数
else:
    decimal_places = 3  # 极小范围（如0.001）：3位小数

# 应用刻度到x轴
ax.set_xticks(ticks)
# 格式化刻度值（按计算的小数位数显示）
format_str = f'%.{decimal_places}f'
ax.set_xticklabels([format_str % x for x in ticks])

# 微调x轴范围（避免刻度贴近图形边缘，优化视觉）
ax.set_xlim(
    new_x_min - (new_x_max - new_x_min) * 0.05,
    new_x_max + (new_x_max - new_x_min) * 0.05
)

# 调整x轴刻度字体大小
for label in ax.get_xticklabels():
    label.set_fontproperties(tick_font)
    label.set_fontsize(tick_font_size)

# 调整x轴标题（使用label_font_size）
ax.set_xlabel('SHAP value', fontproperties=label_font, fontsize=label_font_size)

# 获取颜色条轴（右侧标题所在的轴）
cbar_ax = fig.axes[-1]

# 调整颜色条粗细
cbar_ax.set_aspect(plot_height * 10)

# 调整颜色条上的High和Low标签
for label in cbar_ax.get_yticklabels():
    label.set_fontproperties(colorbar_font)
    label.set_fontsize(colorbar_label_size)

# 重点修改：先清除现有标题再重新设置，确保字体大小生效
if cbar_ax.get_title():
    # 保存当前标题文本
    title_text = cbar_ax.get_title()
    # 清除现有标题
    cbar_ax.set_title('')
    # 重新设置标题，应用字体大小
    cbar_ax.set_title(
        title_text, 
        fontproperties=label_font,
        fontsize=label_font_size,  # 明确设置所需字体大小
        pad=20  # 增加标题与颜色条的间距，避免重叠
    )

# 手动调整边距
plt.subplots_adjust(
    left=0.25,    # 左侧边距
    right=0.94,   # 右侧边距
    top=0.95,     # 顶部边距
    bottom=0.2    # 底部边距
)

# 强制刷新图形
plt.draw()

# 显示图形
plt.show()