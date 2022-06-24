import sys

from matplotlib import pyplot as plt
from tensorflow.keras import mixed_precision

sys.path.append("./")
import tensorflow as tf
from util.data_loader import DataLoader, PoolData

mixed_precision.set_global_policy("mixed_float16")

# 创建构建数据集对象
dataloader = DataLoader(
    train_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_valid_HR",
    test_resource_path="/run/media/zyk/Data/数据集/DIV2K/DIV2K_valid_HR",
    batch_size=4,
    downsample_mode="second-order",
    train_hr_img_height=128,
    train_hr_img_width=128,
    valid_hr_img_height=128,
    valid_hr_img_width=128,
    scale_factor=4,
    max_workers=4,
)

pool_data = PoolData(16, dataloader.batch_size)

mixed_precision.set_global_policy("mixed_float16")

# 加载训练数据集，并训练
for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader.train_data):
    # 若为二阶退化模型，需要先对图像进行退化处理，再从数据池中取出数据
    lr_imgs, hr_imgs = dataloader.feed_second_order_data(
        hr_imgs,
        dataloader.train_hr_img_height,
        dataloader.train_hr_img_width,
        True,
        False,
    )
    lr_imgs, hr_imgs = pool_data.get_pool_data(lr_imgs, hr_imgs)
    print(lr_imgs.dtype, hr_imgs.dtype)

# take_num = 5

# fig, axs = plt.subplots(take_num, 2)
# for i, (_, hr_img) in enumerate(dataloader.train_data.unbatch().take(take_num)):
#     hr_img = tf.expand_dims(hr_img, axis=0)

#     lr_img, hr_img = dataloader.feed_second_order_data(
#         hr_img,
#         dataloader.train_hr_img_height,
#         dataloader.train_hr_img_width,
#         True,
#         False,
#     )
#     lr_img, hr_img = pool_data.get_pool_data(lr_img, hr_img)

#     # 反归一化
#     lr_img, hr_img = (
#         tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8),
#         tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
#     )

#     # 绘制图像
#     axs[i, 0].imshow(lr_img[0])
#     axs[i, 0].axis("off")
#     if i == 0:
#         axs[i, 0].set_title("Second-Order")

#     axs[i, 1].imshow(hr_img[0])
#     axs[i, 1].axis("off")
#     if i == 0:
#         axs[i, 1].set_title("Groud-Truth")

# plt.show()
