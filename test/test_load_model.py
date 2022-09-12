import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         # tf.config.set_logical_device_configuration(
#         #     gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
#         # )
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


sys.path.append("./")
from util.data_loader import DataLoader
from util.niqe import tf_niqe

TAKE_NUM = 3

# 创建构建数据集对象
dataloader = DataLoader(
    train_resource_path="/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5",
    test_resource_path="/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5",
    batch_size=1,
    downsample_mode="bicubic",
    train_hr_img_height=128,
    train_hr_img_width=128,
    valid_hr_img_height=256,
    valid_hr_img_width=256,
    scale_factor=4,
    max_workers=8,
)

# 加载模型
srgan_generator = tf.keras.models.load_model(
    "/run/media/zyk/Data/研究生资料/超分模型结果/srgan/models/train/gen_model_epoch_1000",
    compile=False,
)

esrgan_generator = tf.keras.models.load_model(
    "/run/media/zyk/Data/研究生资料/超分模型结果/esrgan/models/train/gen_model_epoch_6500",
    compile=False,
)

real_esrgan_generator = tf.keras.models.load_model(
    "/home/zyk/下载/gen_model_epoch_20", compile=False
)

rs_esrgan_generator = tf.keras.models.load_model(
    "/run/media/zyk/Data/研究生资料/超分模型结果/rs-esrgan-bicubic/models/train/gen_model_epoch_1000",
    compile=False,
)

# 从测试数据集中取出一批图片
test_dataset = dataloader.test_data.unbatch().skip(0).take(TAKE_NUM)

# 绘图
fig, axs = plt.subplots(TAKE_NUM, 5)
for i, (lr_img, hr_img) in enumerate(test_dataset):
    # 利用生成器生成图片
    srgan_sr_img = srgan_generator.predict(tf.expand_dims(lr_img, 0))
    srgan_sr_img = tf.squeeze(srgan_sr_img, axis=0)

    esrgan_sr_img = esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    esrgan_sr_img = tf.squeeze(esrgan_sr_img, axis=0)

    real_esrgan_img = real_esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    real_esrgan_img = tf.squeeze(real_esrgan_img, axis=0)

    rs_esrgan_sr_img = rs_esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    rs_esrgan_sr_img = tf.squeeze(rs_esrgan_sr_img, axis=0)

    # print(
    #     "NIQE ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}, GT: {:.2f}".format(
    #         tf_niqe(srgan_sr_img),
    #         tf_niqe(esrgan_sr_img),
    #         tf_niqe(rs_esrgan_sr_img),
    #         tf_niqe(hr_img),
    #     )
    # )

    # 反归一化
    lr_img, hr_img, srgan_sr_img, esrgan_sr_img, real_esrgan_img, rs_esrgan_sr_img = (
        tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((srgan_sr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((esrgan_sr_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((real_esrgan_img + 1) * 127.5, dtype=tf.uint8),
        tf.cast((rs_esrgan_sr_img + 1) * 127.5, dtype=tf.uint8),
    )

    print(
        "PSNR ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, REAL-ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}".format(
            tf.image.psnr(hr_img, srgan_sr_img, max_val=255),
            tf.image.psnr(hr_img, esrgan_sr_img, max_val=255),
            tf.image.psnr(hr_img, real_esrgan_img, max_val=255),
            tf.image.psnr(hr_img, rs_esrgan_sr_img, max_val=255),
        )
    )
    print(
        "SSIM ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, REAL-ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}".format(
            tf.image.ssim(hr_img, srgan_sr_img, max_val=255),
            tf.image.ssim(hr_img, esrgan_sr_img, max_val=255),
            tf.image.ssim(hr_img, real_esrgan_img, max_val=255),
            tf.image.ssim(hr_img, rs_esrgan_sr_img, max_val=255),
        )
    )

    axs[i, 0].imshow(lr_img)
    axs[i, 0].axis("off")
    if i == 0:
        axs[i, 0].set_title("Bicubic")

    axs[i, 1].imshow(esrgan_sr_img)
    axs[i, 1].axis("off")
    if i == 0:
        axs[i, 1].set_title("SRGAN")

    axs[i, 2].imshow(esrgan_sr_img)
    axs[i, 2].axis("off")
    if i == 0:
        axs[i, 2].set_title("ESRGAN")

    axs[i, 3].imshow(real_esrgan_img)
    axs[i, 3].axis("off")
    if i == 0:
        axs[i, 3].set_title("REAL-ESRGAN")

    # axs[i, 4].imshow(rs_esrgan_sr_img)
    # axs[i, 4].axis("off")
    # if i == 0:
    #     axs[i, 4].set_title("RS-ESRGAN")

    axs[i, 4].imshow(hr_img)
    axs[i, 4].axis("off")
    if i == 0:
        axs[i, 4].set_title("Groud Truth")

    # 保存图片
    fig.savefig(
        os.path.join("./image/test", "test_3.png"),
        dpi=500,
        bbox_inches="tight",
    )
fig.clear()
plt.close(fig)
