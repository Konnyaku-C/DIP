import numpy as np
import torch
print("NumPy 路径:", np.__file__)
print("PyTorch 路径:", torch.__file__)
# 创建一个 NumPy 数组
foreground_np = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)

# 将 NumPy 数组转换为 PyTorch 张量
fg_img_tensor = torch.from_numpy(foreground_np).permute(2, 0, 1).unsqueeze(0).float() / 255.

print(fg_img_tensor.shape)      # 输出张量形状
print(fg_img_tensor.device)     # 输