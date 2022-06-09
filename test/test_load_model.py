import os

import tensorflow as tf
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 加载模型
generator = tf.keras.models.load_model(
    "./result/esrgan/gen_model_epoch_6500", compile=False
)
# 读取图片
lr_img = tf.io.read_file("./image/test_lr_1.png")
# 解码
lr_img = tf.image.decode_png(lr_img, channels=3)
# 归一化
lr_img = tf.cast(lr_img, tf.float32) / 127.5 - 1

# 升维
lr_img = tf.expand_dims(lr_img, axis=0)
# 进行超分
sr_img = generator(lr_img)
# 降维
sr_img = sr_img[0]
# 反归一化
sr_img = tf.cast((sr_img + 1) * 127.5, dtype=tf.uint8).numpy()

# 保存图片
plt.imsave("./image/test_sr_1.png", sr_img)
