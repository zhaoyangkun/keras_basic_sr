import numpy as np
from tensorflow.keras import backend as K


def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, [in_dim, out_dim])
    u = K.ones([1, in_dim])
    for _ in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))

    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


# 谱归一化层
def spectral_normalization(w):
    return w / spectral_norm(w)
