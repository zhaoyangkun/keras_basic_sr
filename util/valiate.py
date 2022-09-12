import os
import sys
from glob import glob

import tensorflow as tf
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
# from util.data_loader import DataLoader
# from util.lpips import calculate_lpips

TAKE_NUM = 14

# # 创建构建数据集对象
# dataloader = DataLoader(
#     train_resource_path="/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5",
#     test_resource_path="/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5",
#     batch_size=1,
#     downsample_mode="bicubic",
#     train_hr_img_height=128,
#     train_hr_img_width=128,
#     valid_hr_img_height=256,
#     valid_hr_img_width=256,
#     scale_factor=4,
#     max_workers=8,
# )


# 处理图片
def process_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 127.5 - 1  # 归一化到 [-1, 1]
    return image


# 加载图片
def load_image(lr_img_path, hr_img_path):
    lr_img = tf.io.read_file(lr_img_path)
    hr_img = tf.io.read_file(hr_img_path)
    lr_img = process_image(lr_img)
    hr_img = process_image(hr_img)

    return lr_img, hr_img


# 反归一化
def denormalize(image):
    return tf.cast((image + 1) * 127.5, dtype=tf.uint8)


# 获取图片路径
lr_img_resource_path_list = sorted(
    glob(
        os.path.join(
            "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set14/LRbicx4",
            "*[.png]",
        )
    )
)
hr_img_resource_path_list = sorted(
    glob(
        os.path.join(
            "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set14/GTmod12",
            "*[.png]",
        )
    )
)


test_data = (
    tf.data.Dataset.from_tensor_slices(
        (lr_img_resource_path_list, hr_img_resource_path_list)
    )
    .map(
        lambda lr_img_path, hr_img_path: load_image(lr_img_path, hr_img_path),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .batch(1)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# 从测试数据集中取出一批图片
test_dataset = test_data.unbatch().skip(0).take(TAKE_NUM)

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
    "/home/zyk/下载/gen_model_epoch_120",
    compile=False,
)

rs_esrgan_generator = tf.keras.models.load_model(
    "/run/media/zyk/Data/研究生资料/超分模型结果/rs-esrgan-bicubic/models/train/gen_model_epoch_1000",
    compile=False,
)

srgan_psnr_metric = tf.keras.metrics.Mean()
srgan_ssim_metric = tf.keras.metrics.Mean()
srgan_lpips_metric = tf.keras.metrics.Mean()

esrgan_psnr_metric = tf.keras.metrics.Mean()
esrgan_ssim_metric = tf.keras.metrics.Mean()
esrgan_lpips_metric = tf.keras.metrics.Mean()

real_esrgan_psnr_metric = tf.keras.metrics.Mean()
real_esrgan_ssim_metric = tf.keras.metrics.Mean()
real_esrgan_lpips_metric = tf.keras.metrics.Mean()

rs_esrgan_psnr_metric = tf.keras.metrics.Mean()
rs_esrgan_ssim_metric = tf.keras.metrics.Mean()
rs_esrgan_lpips_metric = tf.keras.metrics.Mean()

# 绘图
fig, axs = plt.subplots(TAKE_NUM, 5)
for i, (lr_img, hr_img) in enumerate(test_dataset):
    # 利用生成器生成图片
    srgan_sr_img = srgan_generator.predict(tf.expand_dims(lr_img, 0))
    srgan_sr_img = tf.squeeze(srgan_sr_img, axis=0)

    esrgan_sr_img = esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    esrgan_sr_img = tf.squeeze(esrgan_sr_img, axis=0)

    real_esrgan_sr_img = real_esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    real_esrgan_sr_img = tf.squeeze(real_esrgan_sr_img, axis=0)

    rs_esrgan_sr_img = rs_esrgan_generator.predict(tf.expand_dims(lr_img, 0))
    rs_esrgan_sr_img = tf.squeeze(rs_esrgan_sr_img, axis=0)

    # 反归一化
    (
        lr_img,
        srgan_sr_img,
        esrgan_sr_img,
        real_esrgan_sr_img,
        rs_esrgan_sr_img,
        hr_img,
    ) = (
        denormalize(lr_img),
        denormalize(srgan_sr_img),
        denormalize(esrgan_sr_img),
        denormalize(real_esrgan_sr_img),
        denormalize(rs_esrgan_sr_img),
        denormalize(hr_img),
    )

    # print(
    #     hr_img.shape,
    #     srgan_sr_img.shape,
    #     real_esrgan_sr_img.shape,
    #     rs_esrgan_sr_img.shape,
    # )
    srgan_psnr_metric.update_state(tf.image.psnr(hr_img, srgan_sr_img, max_val=255))
    esrgan_psnr_metric.update_state(tf.image.psnr(hr_img, esrgan_sr_img, max_val=255))
    real_esrgan_psnr_metric.update_state(
        tf.image.psnr(hr_img, real_esrgan_sr_img, max_val=255)
    )
    rs_esrgan_psnr_metric.update_state(
        tf.image.psnr(hr_img, rs_esrgan_sr_img, max_val=255)
    )

    srgan_ssim_metric.update_state(tf.image.ssim(hr_img, srgan_sr_img, max_val=255))
    esrgan_ssim_metric.update_state(tf.image.ssim(hr_img, esrgan_sr_img, max_val=255))
    real_esrgan_ssim_metric.update_state(
        tf.image.ssim(hr_img, real_esrgan_sr_img, max_val=255)
    )
    rs_esrgan_ssim_metric.update_state(
        tf.image.ssim(hr_img, rs_esrgan_sr_img, max_val=255)
    )

    # srgan_lpips_metric.update_state(
    #     calculate_lpips(hr_img.shape[0], hr_img, srgan_sr_img)
    # )
    # esrgan_lpips_metric.update_state(
    #     calculate_lpips(hr_img.shape[0], hr_img, esrgan_sr_img)
    # )
    # real_esrgan_lpips_metric.update_state(
    #     calculate_lpips(hr_img.shape[0], hr_img, real_esrgan_sr_img)
    # )
    # rs_esrgan_lpips_metric.update_state(
    #     calculate_lpips(hr_img.shape[0], hr_img, rs_esrgan_sr_img)
    # )

    axs[i, 0].imshow(lr_img)
    axs[i, 0].axis("off")
    if i == 0:
        axs[i, 0].set_title("LR")

    axs[i, 1].imshow(esrgan_sr_img)
    axs[i, 1].axis("off")
    if i == 0:
        axs[i, 1].set_title("SRGAN")

    axs[i, 2].imshow(esrgan_sr_img)
    axs[i, 2].axis("off")
    if i == 0:
        axs[i, 2].set_title("ESRGAN")

    axs[i, 3].imshow(real_esrgan_sr_img)
    axs[i, 3].axis("off")
    if i == 0:
        axs[i, 3].set_title("REAL-ESRGAN")

    axs[i, 4].imshow(rs_esrgan_sr_img)
    axs[i, 4].axis("off")
    if i == 0:
        axs[i, 4].set_title("RS-ESRGAN")

    # axs[i, 5].imshow(hr_img)
    # axs[i, 5].axis("off")
    # if i == 0:
    #     axs[i, 5].set_title("GT")
# 保存图片
fig.savefig(
    os.path.join("./image/test", "test_4.png"),
    dpi=300,
    bbox_inches="tight",
)
print(
    "PSNR ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, REAL-ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}".format(
        srgan_psnr_metric.result(),
        esrgan_psnr_metric.result(),
        real_esrgan_psnr_metric.result(),
        rs_esrgan_psnr_metric.result(),
    )
)
print(
    "SSIM ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, REAL-ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}".format(
        srgan_ssim_metric.result(),
        esrgan_ssim_metric.result(),
        real_esrgan_ssim_metric.result(),
        rs_esrgan_ssim_metric.result(),
    )
)
# print(
#     "LPIPS ~ SRGAN: {:.2f}, ESRGAN: {:.2f}, REAL-ESRGAN: {:.2f}, RS-ESRGAN: {:.2f}".format(
#         srgan_lpips_metric.result(),
#         esrgan_lpips_metric.result(),
#         real_esrgan_lpips_metric.result(),
#         rs_esrgan_lpips_metric.result(),
#     )
# )
fig.clear()
plt.close(fig)
