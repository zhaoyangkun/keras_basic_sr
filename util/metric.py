import os

import tensorflow as tf
from tensorflow.keras import backend as K

from util.lpips import learned_perceptual_metric_model


# 计算 PSNR
def calculate_psnr(y_true, y_pred):
    # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    return -10 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


# 计算 LPIPS
def calculate_lpips(y_true, y_pred):
    image_size = y_true.shape[1]
    vgg_ckpt = os.path.join("./model", "vgg", "exported")
    lin_ckpt = os.path.join("./model", "lin", "exported")

    lpips_model = learned_perceptual_metric_model(image_size, vgg_ckpt, lin_ckpt)
    metric = lpips_model([y_true, y_pred])

    return metric
    # return tf.reduce_mean(metric)
