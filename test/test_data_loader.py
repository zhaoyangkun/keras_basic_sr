import sys

import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.append("./")
from util.data_loader import DataLoader

# 创建构建数据集对象
dataloader = DataLoader(
    train_resource_path="F:/数据集/DIV2K/DIV2K_valid_HR",
    test_resource_path="F:/数据集/DIV2K/DIV2K_valid_HR",
    batch_size=4,
    downsample_mode="bicubic",
    train_hr_img_height=256,
    train_hr_img_width=256,
    valid_hr_img_height=256,
    valid_hr_img_width=256,
    scale_factor=4,
    max_workers=4,
)

# 显示数据集中的部分图片
take_num = 5

# 绘图
fig, axs = plt.subplots(take_num, 2)
for i, (lr_img, hr_img) in enumerate(dataloader.test_data.unbatch().take(take_num)):
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

# for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader.test_data):
#     # lr_imgs = resize(
#     #     lr_imgs,
#     #     dataloader.train_hr_img_width,
#     #     dataloader.train_hr_img_height,
#     #     3,
#     #     "bicubic",
#     # )
#     # 绘图
#     fig, axs = plt.subplots(lr_imgs.shape[0], 2)
#     for i in range(lr_imgs.shape[0]):
#         lr_img = lr_imgs[i]
#         hr_img = hr_imgs[i]
#         # 反归一化
#         lr_img, hr_img = (
#             tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8),
#             tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
#         )

#         # 绘制图像
#         axs[i, 0].imshow(lr_img)
#         axs[i, 0].axis("off")
#         if i == 0:
#             axs[i, 0].set_title("Bicubic Resize")

#         axs[i, 1].imshow(hr_img)
#         axs[i, 1].axis("off")
#         if i == 0:
#             axs[i, 1].set_title("Groud-Truth")
#     plt.show()
#     break
