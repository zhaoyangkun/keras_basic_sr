import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf

hr_height = 128
# patch = hr_height // 2**4
# dis_patch = (patch, patch, 1)
# # print(patch)

# real_labels = tf.ones((4,) + dis_patch)
# print(real_labels)


def build_vgg_1():
    """
    构建 vgg 模型
    """
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    model = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)
    model.trainable = False
    # vgg.trainable = False
    # for l in vgg.layers:
    #     l.trainable = False
    return model


def build_vgg_2():
    # 创建VGG模型，只使用第 20 层的特征
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    model = Model(vgg.input, vgg.layers[20].output)
    model.trainable = False

    return model


vgg_1 = build_vgg_2()
vgg_2 = build_vgg_2()

fake_img = np.ones((1, hr_height, hr_height, 3))
features_1 = vgg_1.predict(fake_img)
features_2 = vgg_2.predict(fake_img)
assert (features_1.all() == features_2.all())
