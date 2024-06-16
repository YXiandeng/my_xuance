import matplotlib.pyplot as plt
# 读取txt文件
with open('settings/u_loc2.txt', 'r') as f:
    lines = f.readlines()

# 将坐标数据分离出来
x_values = []
y_values = []
for line in lines:
    x, y = line.split()
    x_values.append(float(x))
    y_values.append(float(y))

# 绘制散点图
plt.scatter(x_values, y_values)
plt.show()