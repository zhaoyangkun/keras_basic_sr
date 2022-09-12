import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt


sys.path.append("./")
from util.lpips import calculate_lpips

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 加载并处理图片
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 127.5 - 1  # 归一化到 [-1, 1]
    return image


# 反归一化
def denormalize(image):
    return tf.cast((image + 1) * 127.5, dtype=tf.uint8)


# 加载模型
esrgan_generator = tf.keras.models.load_model(
    "/run/media/zyk/Data/研究生资料/超分模型结果/esrgan/models/train/gen_model_epoch_6500",
    compile=False,
)

real_esrgan_generator = tf.keras.models.load_model(
    "/home/zyk/下载/gen_model_epoch_80", compile=False
)

# rs_esrgan_generator = tf.keras.models.load_model(
#     "/run/media/zyk/Data/研究生资料/超分模型结果/rs-esrgan-bicubic/models/train/gen_model_epoch_1000",
#     compile=False,
# )

# 加载低分辨率图像
lr_img = load_and_preprocess_image(
    "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set/LRbicx4/baby.png"
)
hr_img = load_and_preprocess_image(
    "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5/GTmod12/baby.png"
)


# 生成超分图像
esrgan_hr_img = esrgan_generator.predict(tf.expand_dims(lr_img, 0))
real_esrgan_hr_img = real_esrgan_generator.predict(tf.expand_dims(lr_img, 0))
# rs_esrgan_hr_img = rs_esrgan_generator.predict(tf.expand_dims(lr_img, 0))

esrgan_hr_img = tf.squeeze(esrgan_hr_img, axis=0)
real_esrgan_hr_img = tf.squeeze(real_esrgan_hr_img, axis=0)
# rs_esrgan_hr_img = tf.squeeze(rs_esrgan_hr_img, axis=0)

lr_img, esrgan_hr_img, real_esrgan_hr_img, hr_img = (
    denormalize(lr_img),
    denormalize(esrgan_hr_img),
    denormalize(real_esrgan_hr_img),
    denormalize(hr_img),
)

print(
    "psnr ~ esrgan: {:.2f}, real-esrgan: {:.2f}".format(
        tf.image.psnr(esrgan_hr_img, hr_img, max_val=255),
        tf.image.psnr(real_esrgan_hr_img, hr_img, max_val=255),
    )
)
print(
    "ssim ~ esrgan: {:.2f}, real-esrgan: {:.2f}".format(
        tf.image.ssim(esrgan_hr_img, hr_img, max_val=255),
        tf.image.ssim(real_esrgan_hr_img, hr_img, max_val=255),
    )
)

# print(hr_img.shape, esrgan_hr_img.shape, real_esrgan_hr_img.shape)

print(
    "lpips ~ esrgan: {:.2f}, real-esrgan: {:.2f}".format(
        calculate_lpips(
            hr_img.shape[0],
            tf.expand_dims(hr_img, 0),
            tf.expand_dims(esrgan_hr_img, 0),
        ),
        calculate_lpips(
            hr_img.shape[0],
            tf.expand_dims(hr_img, 0),
            tf.expand_dims(real_esrgan_hr_img, 0),
        ),
    )
)

# 绘图
fig, axs = plt.subplots(1, 3)
axs[0].imshow(lr_img)
axs[0].axis("off")
axs[0].set_title("LR")

axs[1].imshow(esrgan_hr_img)
axs[1].axis("off")
axs[1].set_title("ESRGAN")

axs[2].imshow(real_esrgan_hr_img)
axs[2].axis("off")
axs[2].set_title("Real-ESRGAN")

plt.show()
