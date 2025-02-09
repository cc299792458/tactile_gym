import numpy as np
import matplotlib.pyplot as plt

# 加载 npy 文件
file_path = '/home/chichu/Documents/tactile_gym/tactile_gym/assets/robot_assets/tactip/reference_images/flat/256x256/nodef_gray.npy'  # 你的 np 文件路径
data = np.load(file_path)

# 如果是多通道数据，确保格式为 (H, W, C)
if data.ndim == 2:  # 灰度图
    plt.imshow(data, cmap="gray")
elif data.ndim == 3:  # RGB 图像
    plt.imshow(data)
else:
    raise ValueError("Unsupported image shape:", data.shape)

plt.axis("off")  # 去掉坐标轴
plt.show()
