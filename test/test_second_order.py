import sys

from matplotlib import pyplot as plt
sys.path.append("./")
from util.toml import parse_toml

import tensorflow as tf
from util.data_loader import DataLoader

# 创建构建数据集对象
dataloader = DataLoader(
    train_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_train_HR",
    test_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_valid_HR",
    batch_size=4,
    downsample_mode="second-order",
    train_hr_img_height=256,
    train_hr_img_width=256,
    valid_hr_img_height=256,
    valid_hr_img_width=256,
    scale_factor=4,
    max_workers=4,
)

# pool_data = PoolData(16, dataloader.batch_size)

config = parse_toml("./config/config.toml")
degration_config = config["second-order-degradation"]
take_num = 4
fig, axs = plt.subplots(take_num, 4)
for i, (lr_img, hr_img) in enumerate(dataloader.test_data.unbatch().skip(2).take(take_num)):
    lr_img_bicubic = tf.expand_dims(lr_img, axis=0)
    hr_img = tf.expand_dims(hr_img, axis=0)

    lr_img_second_order, usm_hr_img = dataloader.feed_second_order_data(
        (hr_img + 1) / 2,
        degration_config,
        dataloader.train_hr_img_height,
        dataloader.train_hr_img_width,
        False,
        False,
        True,
    )
    # lr_img_second_order, usm_hr_img = pool_data.get_pool_data(lr_img, hr_img)

    # 反归一化
    lr_img_bicubic, lr_img_second_order, usm_hr_img, hr_img = (
        tf.cast((lr_img_bicubic + 1) * 127.5, dtype=tf.uint8),
        tf.cast((lr_img_second_order + 1) * 127.5, dtype=tf.uint8),
        tf.cast((usm_hr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
    )

    # 绘制图像
    axs[i, 0].imshow(lr_img_bicubic[0])
    axs[i, 0].axis("off")
    if i == 0:
        axs[i, 0].set_title("Bicubic")

    axs[i, 1].imshow(lr_img_second_order[0])
    axs[i, 1].axis("off")
    if i == 0:
        axs[i, 1].set_title("Second-Order")

    axs[i, 2].imshow(usm_hr_img[0])
    axs[i, 2].axis("off")
    if i == 0:
        axs[i, 2].set_title("USM-HR")

    axs[i, 3].imshow(hr_img[0])
    axs[i, 3].axis("off")
    if i == 0:
        axs[i, 3].set_title("HR")
plt.savefig("test_second_order.png", dpi=300)
# plt.show()
