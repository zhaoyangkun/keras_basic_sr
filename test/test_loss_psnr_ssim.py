import numpy as np
from matplotlib import pyplot as plt

epoch_list = np.arange(0, 10 * 3, 1)
g_loss_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 3
d_loss_list = [0.2, 0.9, 0.4, 0.3, 0.4, 0.3, 0.7, 0.4, 0.7, 0.9] * 3
psnr_list = [20, 30, 21, 19, 23, 26, 28, 29, 30, 32] * 3
ssim_list = [0.2, 0.3, 0.21, 0.19, 0.23, 0.26, 0.28, 0.29, 0.3, 0.56] * 3

fig = plt.figure(figsize=(10, 10))

# 绘制损失曲线
ax_1 = plt.subplot(2, 1, 1)
ax_1.set_title("Loss")
(line_1,) = ax_1.plot(
    epoch_list, g_loss_list, color="deepskyblue", marker=".", label="g_loss"
)
(line_2,) = ax_1.plot(
    epoch_list, d_loss_list, color="darksalmon", marker=".", label="d_loss"
)
ax_1.set_xlabel("epoch")
ax_1.set_ylabel("Loss")
ax_1.legend(handles=[line_1, line_2], loc="upper right")

# 绘制 PSNR 曲线
ax_2 = plt.subplot(2, 2, 3)
ax_2.set_title("PSNR")
(line_3,) = ax_2.plot(epoch_list, psnr_list, color="orange", marker=".", label="PSNR")
ax_2.set_xlabel("epoch")
ax_2.set_ylabel("PSNR")
ax_2.legend(handles=[line_3], loc="upper right")

# 绘制损失曲线
ax_3 = plt.subplot(2, 2, 4)
ax_3.set_title("SSIM")
(line_4,) = ax_3.plot(epoch_list, ssim_list, color="skyblue", marker=".", label="SSIM")
ax_3.set_xlabel("epoch")
ax_3.set_ylabel("SSIM")
ax_3.legend(handles=[line_4], loc="upper right")

fig.tight_layout()
plt.show()
