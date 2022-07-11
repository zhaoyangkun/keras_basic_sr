import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (Add, Concatenate, Conv2D, Input, Lambda,
                                     LeakyReLU, MaxPooling2D, PReLU, ReLU,
                                     UpSampling2D)
from tensorflow_addons.layers import SpectralNormalization


# 基于谱归一化的卷积层
def spectral_norm_conv2d(input, use_sn=True, sn_dtype=None, **kwargs):
    if use_sn:
        if sn_dtype:
            return SpectralNormalization(Conv2D(**kwargs, dtype=sn_dtype), dtype=sn_dtype)(input)
        return SpectralNormalization(Conv2D(**kwargs))(input)
    else:
        return Conv2D(**kwargs)(input)

def subpixel_conv2d(name, scale=2):
    """亚像素卷积层

    Args:
        name (str): 名称
        scale (int, optional): 缩放比例. 默认为 2.
    """

    def subpixel_shape(input_shape):
        dims = [
            input_shape[0],
            None if input_shape[1] is None else input_shape[1] * scale,
            None if input_shape[2] is None else input_shape[2] * scale,
            int(input_shape[3] / (scale**2)),
        ]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)

# 上采样
def upsample(x, number, channels=64):
    x = Conv2D(
        channels * 4,
        kernel_size=3,
        strides=1,
        padding="same",
        name="up_sample_conv2d_" + str(number),
    )(x)
    x = subpixel_conv2d("up_sample_subpixel_" + str(number), 2)(x)
    x = PReLU(shared_axes=[1, 2], name="up_sample_prelu_" + str(number))(x)

    return x

# 交替上采样 + RFB
def upsample_rfb(x, number, method="nearest", channels=64):
    # 最近邻域插值上采样
    if method == "nearest":
        x = UpSampling2D(
            size=2,
            interpolation="nearest",
            name="up_sample_nearest_" + str(number),
        )(x)
        x = RFB(x, in_channels=channels, out_channels=channels)
        x = LeakyReLU(0.2)(x)
    # 双线性插值上采样
    elif method == "bilinear":
        x = UpSampling2D(
            size=2,
            interpolation="bilinear",
            name="up_sample_bilinear_" + str(number),
        )(x)
        x = RFB(x, in_channels=channels, out_channels=channels)
        x = LeakyReLU(0.2)(x)
    # 亚像素卷积上采样
    elif method == "subpixel":
        x = Conv2D(
            channels * 4,
            kernel_size=3,
            strides=1,
            padding="same",
            name="up_sample_conv2d_" + str(number),
        )(x)
        x = subpixel_conv2d("up_sample_subpixel_" + str(number), 2)(x)
        x = RFB(x, in_channels=channels, out_channels=channels)
        x = LeakyReLU(0.2)(x)
    else:
        raise ValueError("Unsupported upsample method!")

    return x

# 构建 dense block
def dense_block(input):
    x1 = Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
    )(input)
    x1 = LeakyReLU(0.2)(x1)

    x2 = Concatenate()([input, x1])
    x2 = Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x2)
    x2 = LeakyReLU(0.2)(x2)

    x3 = Concatenate()([input, x1, x2])
    x3 = Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x3)
    x3 = LeakyReLU(0.2)(x3)

    x4 = Concatenate()([input, x1, x2, x3])
    x4 = Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x4)
    x4 = LeakyReLU(0.2)(x4)

    x5 = Concatenate()([input, x1, x2, x3, x4])
    x5 = Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x5)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    output = Add()([x5, input])

    return output

# 构建 RRDB
def RRDB(input):
    x = dense_block(input)
    x = dense_block(x)
    x = dense_block(x)
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])

    return out

# 构建 RFB
def RFB(input, in_channels=64, out_channels=32):
    branch_channels = in_channels // 4

    shortcut = Conv2D(
        out_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    shortcut = Lambda(lambda x: x * 0.2)(shortcut)

    # 分支 1
    x_1 = Conv2D(
        branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_1 = LeakyReLU(0.2)(x_1)
    x_1 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x_1)

    # 分支 2
    x_2 = Conv2D(
        branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_2 = LeakyReLU(0.2)(x_2)
    x_2 = Conv2D(
        branch_channels,
        kernel_size=(1, 3),
        strides=1,
        padding="same",
    )(x_2)
    x_2 = LeakyReLU(0.2)(x_2)
    x_2 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        dilation_rate=3,
        padding="same",
    )(x_2)

    # 分支 3
    x_3 = Conv2D(
        branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_3 = LeakyReLU(0.2)(x_3)
    x_3 = Conv2D(
        branch_channels,
        kernel_size=(3, 1),
        strides=1,
        padding="same",
    )(x_3)
    x_3 = LeakyReLU(0.2)(x_3)
    x_3 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        dilation_rate=3,
        padding="same",
    )(x_3)

    # 分支 4
    x_4 = Conv2D(
        branch_channels // 2,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_4 = LeakyReLU(0.2)(x_4)
    x_4 = Conv2D(
        (branch_channels // 4) * 3,
        kernel_size=(1, 3),
        strides=1,
        padding="same",
    )(x_4)
    x_4 = LeakyReLU(0.2)(x_4)
    x_4 = Conv2D(
        branch_channels,
        kernel_size=(1, 3),
        strides=1,
        padding="same",
    )(x_4)
    x_4 = LeakyReLU(0.2)(x_4)
    x_4 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        dilation_rate=5,
        padding="same",
    )(x_4)

    output = Concatenate()([x_1, x_2, x_3, x_4])
    output = Conv2D(
        out_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(output)
    output = Add()([output, shortcut])

    return output

# 构建 RFDB
def RFDB(input, in_channels=64, growth_channels=32):
    x_1 = RFB(
        input,
        in_channels=in_channels,
        out_channels=growth_channels,
    )
    x_1 = LeakyReLU(0.2)(x_1)

    x_2 = Concatenate()([input, x_1])
    x_2 = RFB(
        x_2,
        in_channels=in_channels + growth_channels,
        out_channels=growth_channels,
    )
    x_2 = LeakyReLU(0.2)(x_2)

    x_3 = Concatenate()([input, x_1, x_2])
    x_3 = RFB(
        x_3,
        in_channels=in_channels + growth_channels * 2,
        out_channels=growth_channels,
    )
    x_3 = LeakyReLU(0.2)(x_3)

    x_4 = Concatenate()([input, x_1, x_2, x_3])
    x_4 = RFB(
        x_4,
        in_channels=in_channels + growth_channels * 3,
        out_channels=growth_channels,
    )
    x_4 = LeakyReLU(0.2)(x_4)

    x_5 = Concatenate()([input, x_1, x_2, x_3, x_4])
    x_5 = RFB(
        x_5,
        in_channels=in_channels + growth_channels * 4,
        out_channels=in_channels,
    )
    x_5 = Lambda(lambda x: x * 0.2)(x_5)
    output = Add()([x_5, input])

    return output

# 构建 RRFDB
def RRFDB(input, input_channels=64, growth_channels=32):
    x = RFDB(input, input_channels, growth_channels)
    x = RFDB(x, input_channels, growth_channels)
    x = RFDB(x, input_channels, growth_channels)

    x = Lambda(lambda x: x * 0.2)(x)
    output = Add()([x, input])

    return output

# 构建 VGG19 模型
def create_vgg19_custom_model():
    # Block 1
    input = Input(shape=(None, None, 3))
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1")(input)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv2", dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding="same", name="block2_conv2", dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv4", dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv4", dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv4", dtype="float32")(x)
    x = ReLU(name="block5_conv4_relu", dtype="float32")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    return Model(input, x, name="vgg19_features")

# 构建 vgg_19 特征提取模型
def create_vgg_19_features_model(loss_type="srgan"):
    original_vgg = VGG19(
        weights="imagenet",
        include_top=False,
    )
    vgg_model = create_vgg19_custom_model()
    vgg_model.set_weights(original_vgg.get_weights())
    
    if loss_type == "srgan":
        outputs = vgg_model.get_layer("block5_conv4_relu").output
    elif loss_type == "esrgan":
        outputs = vgg_model.get_layer("block5_conv4").output
    elif loss_type == "real-esrgan":
        layers = [
            "block1_conv2",
            "block2_conv2",
            "block3_conv4",
            "block4_conv4",
            "block5_conv4",
        ]
        outputs = [vgg_model.get_layer(name).output for name in layers]

    model = Model([vgg_model.input], outputs)
    
    model.trainable = False
    
    return model


# # 谱归一化层
# class SpectralNorm(tf.keras.constraints.Constraint):
#     def __init__(self, n_iter=5):
#         self.n_iter = n_iter

#     def call(self, input_weights):
#         w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
#         u = tf.random.normal((w.shape[0], 1))
#         for _ in range(self.n_iter):
#             v = tf.matmul(w, u, transpose_a=True)
#             v /= tf.norm(v)
#             u = tf.matmul(w, v)
#             u /= tf.norm(u)
#         spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)
#         return input_weights / spec_norm
    