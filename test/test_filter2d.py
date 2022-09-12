import sys

import tensorflow as tf
from tensorflow.keras import layers

sys.path.append("./")
from util.data_util import generate_kernel


def filter2D(imgs, kernels):
    b, h, w, c = imgs.shape
    k = kernels.shape[-1]
    pad_size = k // 2

    # 边缘填充
    padding = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
    imgs = tf.pad(imgs, padding, "REFLECT")

    _, ph, pw, _ = imgs.shape

    # 调整图像的维度
    imgs = tf.transpose(imgs, [1, 2, 3, 0])  # H x W x C x B
    imgs = tf.reshape(imgs, (1, ph, pw, c * b))  # 1 x H x W x B*C

    # 调整卷积核的维度
    kernels = tf.transpose(kernels, [1, 2, 0])  # k, k, b
    kernels = tf.reshape(kernels, [k, k, 1, b])  # k, k, 1, b
    kernels = tf.repeat(kernels, repeats=[c], axis=-1)  # k, k, 1, b * c

    # kernel_height, kernel_width, input_filters, output_filters
    conv = layers.Conv2D(b * c, k, weights=[kernels], use_bias=False, groups=b * c)
    conv.trainable = False

    # 对图像进行卷积
    imgs = conv(imgs)

    # 调整图像的维度
    imgs = tf.reshape(imgs, (h, w, c, b))  # H x W x C x B
    imgs = tf.transpose(imgs, [3, 0, 1, 2])  # B x H x W x C

    return imgs


kernel_props_1 = {
    "kernel_list": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sigma_x_range": [0.2, 3],
    "sigma_y_range": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "sinc_prob": 0.1,
}

imgs = tf.random.normal([1, 32, 32, 3])
first_blur_kernel = generate_kernel(kernel_props_1)
first_blur_kernel = tf.expand_dims(first_blur_kernel, axis=0)
filter2D(imgs, first_blur_kernel)
