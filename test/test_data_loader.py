import sys

import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.append("./")
from util.data_loader import DataLoader

# 创建构建数据集对象
dataloader = DataLoader(
    train_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_train_HR",
    test_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_valid_HR",
    batch_size=4,
    downsample_mode="bicubic",
    train_hr_img_height=128,
    train_hr_img_width=128,
    valid_hr_img_height=128,
    valid_hr_img_width=128,
    scale_factor=4,
    max_workers=4,
)

# for i, (lr_img, hr_img) in enumerate(dataloader.test_data.unbatch().take(2)):
#     print(lr_img.shape, hr_img.shape)

# test_dataset = dataloader.test_data.unbatch().take(10)
# print(test_dataset)

# 显示数据集中的部分图片
# plt.suptitle("test_data")
# dataloader.test_data.skip(5).unbatch().take(6)

take_num = 5
# 绘图
fig, axs = plt.subplots(take_num, 2)
for i, (lr_img, hr_img) in enumerate(dataloader.train_data.unbatch().take(take_num)):
    # 反归一化
    lr_img, hr_img = (
        tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
    )

    # 绘制图像
    axs[i, 0].imshow(lr_img)
    axs[i, 0].axis("off")
    if i == 0:
        axs[i, 0].set_title("Bicubic")

    axs[i, 1].imshow(hr_img)
    axs[i, 1].axis("off")
    if i == 0:
        axs[i, 1].set_title("Groud-Truth")

plt.show()
