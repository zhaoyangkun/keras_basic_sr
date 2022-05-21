from tensorflow.keras import backend as K

def psnr(y_true, y_pred):
    # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    return -10 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)