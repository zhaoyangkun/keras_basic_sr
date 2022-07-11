import tensorflow as tf

# # 计算 PSNR
# def calculate_psnr(y_true, y_pred):
#     # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
#     return -10 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

# 计算 SSIM
def calculate_ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


# 计算 PSNR
def calculate_psnr(y_true, y_pred):
    tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
