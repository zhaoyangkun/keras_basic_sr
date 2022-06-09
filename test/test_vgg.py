import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf

hr_height = 128
patch = hr_height // 2 ** 4
dis_patch = (patch, patch, 1)
# print(patch)

real_labels = tf.ones((4,) + dis_patch)
print(real_labels)


def build_vgg():
    """
    构建 vgg 模型
    """
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    vgg.trainable = False
    for l in vgg.layers:
        l.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)


def build_vgg_2():
    # 创建VGG模型，只使用第9层的特征
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    img_features = [vgg.layers[9].output]
    return Model(vgg.input, img_features)


vgg = build_vgg_2()
vgg.summary()
fake_img = np.ones((1, 128, 128, 3))
fake_img_features = vgg.predict(fake_img)
print(fake_img_features.shape)
