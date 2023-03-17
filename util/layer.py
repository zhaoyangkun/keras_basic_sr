import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (Activation, Add, AvgPool2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, Lambda,
                                     LeakyReLU, MaxPooling2D, Multiply, PReLU,
                                     ReLU, Reshape, UpSampling2D, Layer)
from tensorflow_addons.layers import SpectralNormalization


# 基于谱归一化的卷积层
def spectral_norm_conv2d(input, use_sn=True, sn_dtype=None, **kwargs):
    if use_sn:
        if sn_dtype:
            return SpectralNormalization(Conv2D(**kwargs, dtype=sn_dtype),
                                         dtype=sn_dtype)(input)
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
    # 双线性插值上采样
    elif method == "bilinear":
        x = UpSampling2D(
            size=2,
            interpolation="bilinear",
            name="up_sample_bilinear_" + str(number),
        )(x)
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
    else:
        raise ValueError("Unsupported upsample method!")

    x = RFB(x, in_channels=channels, out_channels=channels)
    x = LeakyReLU(0.2)(x)

    return x


# 交替上采样 + MHARB
def upsample_mharb(x, number, method="nearest", channels=64):
    # 最近邻域插值上采样
    if method == "nearest":
        x = UpSampling2D(
            size=2,
            interpolation="nearest",
            name="up_sample_nearest_" + str(number),
        )(x)
    # 双线性插值上采样
    elif method == "bilinear":
        x = UpSampling2D(
            size=2,
            interpolation="bilinear",
            name="up_sample_bilinear_" + str(number),
        )(x)
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
    else:
        raise ValueError("Unsupported upsample method!")

    x = MHARB(x, in_channels=channels, out_channels=channels)
    x = LeakyReLU(alpha=0.2)(x)

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


# 通道注意力
def channel_attention(input, ratio=8):
    channels = input.shape[-1]

    max_pool = GlobalMaxPooling2D()(input)
    # (batch, channels)
    max_pool = Reshape((1, 1, channels))(max_pool)
    # (batch, 1, 1, channels)
    max_pool = Conv2D(channels // ratio,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      activation="relu")(max_pool)
    # (batch, 1, 1, channels // ratio)
    max_pool = Conv2D(channels, kernel_size=1, strides=1,
                      padding="same")(max_pool)
    # (batch, 1, 1, channels)

    avg_pool = GlobalAveragePooling2D()(input)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    avg_pool = Conv2D(channels // ratio,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      activation="relu")(avg_pool)
    avg_pool = Conv2D(channels, kernel_size=1, strides=1,
                      padding="same")(avg_pool)
    # (batch, 1, 1, channels)

    feature = Add()([max_pool, avg_pool])  # (batch, 1, 1, channels)
    feature = Activation("sigmoid")(feature)  # (batch, 1, 1, channels)

    # x = Multiply()([input, feature])  # (batch, height, width, channels)

    return feature


# 坐标注意力
def coordinate_attention(input, reduction=32, use_bn=False):

    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        return x * tmpx

    _, h, w, c = input.shape
    x_h = AvgPool2D(pool_size=(1, w), strides=1,
                    padding="same")(input)  # [b, h, 1, c]
    x_w = AvgPool2D(pool_size=(h, 1), strides=1,
                    padding="same")(input)  # [b, 1, w, c]
    x_w = tf.transpose(x_w, [0, 2, 1, 3])  # [b, w, 1, c]

    y = tf.concat([x_h, x_w], axis=1)  # [b, h+w, 1, c]
    mip = max(8, c // reduction)
    y = Conv2D(mip, 1, strides=1, padding="same")(y)  # [b, h+w, 1, mip]
    if use_bn:
        y = BatchNormalization()(y)
    y = coord_act(y)

    x_h, x_w = tf.split(y, num_or_size_splits=[h, w],
                        axis=1)  # [b, h, 1, mip], [b, w, 1, mip]
    x_w = tf.transpose(x_w, [0, 2, 1, 3])  # [b, 1, w, mip]
    a_h = Conv2D(c, 1, strides=1, padding="same",
                 activation="sigmoid")(x_h)  # [b, h, 1, c]
    a_w = Conv2D(c, 1, strides=1, padding="same",
                 activation="sigmoid")(x_w)  # [b, 1, w, c]
    output = input * a_h * a_w  # [b, h, w, c]

    return output


# 空间注意力
def spatial_attention(input, in_channels=64, out_channels=32):
    branch_channels = in_channels // 2

    # 分支 1
    x_1 = Conv2D(
        branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_1 = LeakyReLU(alpha=0.2)(x_1)
    x_1 = Conv2D(
        branch_channels,
        kernel_size=(1, 3),
        strides=1,
        padding="same",
    )(x_1)
    x_1 = LeakyReLU(alpha=0.2)(x_1)
    x_1 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        dilation_rate=3,
        padding="same",
    )(x_1)  # (b, h, w, branch_channels)

    # 分支 2
    x_2 = Conv2D(
        branch_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(input)
    x_2 = LeakyReLU(alpha=0.2)(x_2)
    x_2 = Conv2D(
        branch_channels,
        kernel_size=(3, 1),
        strides=1,
        padding="same",
    )(x_2)
    x_2 = LeakyReLU(alpha=0.2)(x_2)
    x_2 = Conv2D(
        branch_channels,
        kernel_size=3,
        strides=1,
        dilation_rate=5,
        padding="same",
    )(x_2)  # (b, h, w, branch_channels)

    output = Concatenate()([x_1, x_2])  # (b, h, w, branch_channels * 2)
    output = Add()([output, input])  # (b, h, w, branch_channels * 2)
    output = Conv2D(
        out_channels,
        kernel_size=1,
        strides=1,
        padding="same",
    )(output)  # (b, h, w, in_channels)

    return output


# 构建 MHARB
def MHARB(input, in_channels=64, out_channels=32, structure="serial"):
    # 跳跃连接
    shortcut = Conv2D(out_channels, 3, strides=1, padding="same")(input)

    # # 自注意力
    # self_feature = SelfAttention(in_dims=in_channels)(input)
    # self_feature = Conv2D(out_channels, 3, strides=1,
    #                       padding="same")(self_feature)

    if structure == "serial":  # 串联
        # 利用空间注意力模块来提取空间信息
        spatial_feature = spatial_attention(
            input, in_channels=in_channels,
            out_channels=out_channels)  # (b, h, w, out_channels)

        # 利用通道注意力模块中提取通道信息
        channel_feature = channel_attention(
            spatial_feature)  # (b, 1, 1, out_channels)

        # 将空间信息和通道信息相乘
        x = Multiply()([spatial_feature,
                        channel_feature])  # (b, h, w, out_channels)
    elif structure == "parallel":  # 并联
        # 利用空间注意力模块来提取空间信息
        spatial_feature = spatial_attention(
            input, in_channels=in_channels,
            out_channels=in_channels)  # (b, h, w, in_channels)

        # 利用通道注意力模块中提取通道信息
        channel_feature = channel_attention(input)  # (b, 1, 1, in_channels)

        # 将空间信息和通道信息相乘
        x = Multiply()([spatial_feature,
                        channel_feature])  # (b, h, w, in_channels)

        # 调整通道维度
        x = Conv2D(out_channels, 3, strides=1, padding="same")(x)
    else:
        raise ValueError("Structure must be 'serial' or 'parallel'!")

    output = Add()([x, shortcut])

    return output


# 构建 MHARG
def MHARG(input, in_channels=64, out_channels=64):
    x = MHARB(input, in_channels, out_channels)
    x = MHARB(x, in_channels, out_channels)
    x = MHARB(x, in_channels, out_channels)
    x = Lambda(lambda x: x * 0.2)(x)

    output = Add()([x, input])

    return output


# 构建 VGG19 模型
def create_vgg19_custom_model():
    # Block 1
    input = Input(shape=(None, None, 3))
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1")(input)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3),
               padding="same",
               name="block1_conv2",
               dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3),
               padding="same",
               name="block2_conv2",
               dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3),
               padding="same",
               name="block3_conv4",
               dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3),
               padding="same",
               name="block4_conv4",
               dtype="float32")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv1")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv2")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv3")(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3),
               padding="same",
               name="block5_conv4",
               dtype="float32")(x)
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
        # outputs = vgg_model.get_layer("block5_conv4").output
        outputs = vgg_model.get_layer("block5_conv4_relu").output
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


# EMA
class EMA:

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    # 注册所有需要跟踪的变量
    def register(self):
        for param in self.model.variables:
            if param.trainable:
                self.shadow[param.name] = param.value()

    # 每次变量的值改变后更新影子变量的值
    def update(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                new_average = (1.0 - self.decay) * param.value(
                ) + self.decay * self.shadow[param.name]
                self.shadow[param.name] = new_average

    # 将模型参数变成影子值，backup是真实值的备份
    def apply_shadow(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                self.backup[param.name] = param.value()
                param.assign(self.shadow[param.name])

    # 将模型的参数变回真实值
    def restore(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.backup
                param.assign(self.backup[param.name])
        self.backup = {}


class SelfAttention(Model):
    """
    自注意力
    """

    def __init__(self, in_dims, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.in_channels = in_dims
        self.query_conv = Conv2D(in_dims // 8, 1)
        self.key_conv = Conv2D(in_dims // 8, 1)
        self.value_conv = Conv2D(in_dims, 1)

    def build(self, input_shape):
        self.gamma = self.add_weight(self.name + "_gamma",
                                     shape=(),
                                     initializer=tf.initializers.zeros)

    def call(self, inputs):
        batch_size, height, width, channels = inputs.shape

        proj_query = self.query_conv(inputs)  # [b, h, w, in_dims // 8]
        proj_query = tf.reshape(
            proj_query,
            (batch_size, width * height, -1))  # [b, h*w, in_dims // 8]

        proj_key = self.key_conv(inputs)  # [b, h, w, in_dims // 8]
        proj_key = tf.transpose(tf.reshape(proj_key,
                                           (batch_size, width * height, -1)),
                                perm=[0, 2, 1])  # [b, in_dims // 8, h*w]

        energy = tf.matmul(proj_query, proj_key)  # [b, h*w, h*w]
        attention = tf.nn.softmax(energy)  # [b, h*w, h*w]

        proj_value = self.value_conv(inputs)  # [b, h, w, in_dims]
        proj_value = tf.transpose(tf.reshape(proj_value,
                                             (batch_size, width * height, -1)),
                                  perm=[0, 2, 1])  # [b, in_dims, h*w]

        out = tf.matmul(proj_value, attention)  # [b, in_dims, h*w]
        out = tf.reshape(
            tf.transpose(out, perm=[0, 2, 1]),
            (batch_size, height, width, channels))  # [b, h, w, in_dims]

        return tf.add(tf.multiply(self.gamma, out), inputs)


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
