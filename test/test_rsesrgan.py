import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    PReLU,
)
from tensorflow.keras.models import Model


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


def build_generator():
    """
    构建生成器
    """

    def upsample(x, number):
        x = Conv2D(
            256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="up_sample_conv2d_" + str(number),
        )(x)
        x = subpixel_conv2d("up_sample_subpixel_" + str(number), 2)(x)
        x = PReLU(shared_axes=[1, 2], name="up_sample_prelu_" + str(number))(x)

        return x

    def dense_block(input):
        x1 = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(input)
        x1 = LeakyReLU(0.2)(x1)
        x1 = Concatenate()([input, x1])

        x2 = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x1)
        x2 = LeakyReLU(0.2)(x2)
        x2 = Concatenate()([input, x1, x2])

        x3 = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x2)
        x3 = LeakyReLU(0.2)(x3)
        x3 = Concatenate()([input, x1, x2, x3])

        x4 = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x3)
        x4 = LeakyReLU(0.2)(x4)
        x4 = Concatenate()([input, x1, x2, x3, x4])

        x5 = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x4)
        x5 = Lambda(lambda x: x * 0.2)(x5)
        output = Add()([x5, input])

        return output

    def RRDB(input):
        x = dense_block(input)
        x = dense_block(x)
        x = dense_block(x)
        x = Lambda(lambda x: x * 0.2)(x)
        out = Add()([x, input])

        return out

    def RFB(input, in_channels=64, out_channels=32):
        branch_channels = in_channels // 4

        shortcut = Conv2D(out_channels, kernel_size=1, strides=1, padding="same")(input)
        shortcut = Lambda(lambda x: x * 0.2)(shortcut)

        # 分支 1
        x_1 = Conv2D(branch_channels, kernel_size=1, strides=1, padding="same")(input)
        x_1 = LeakyReLU(0.2)(x_1)
        x_1 = Conv2D(branch_channels, kernel_size=3, strides=1, padding="same")(x_1)

        # 分支 2
        x_2 = Conv2D(branch_channels, kernel_size=1, strides=1, padding="same")(input)
        x_2 = LeakyReLU(0.2)(x_2)
        x_2 = Conv2D(branch_channels, kernel_size=(1, 3), strides=1, padding="same")(
            x_2
        )
        x_2 = LeakyReLU(0.2)(x_2)
        x_2 = Conv2D(
            branch_channels,
            kernel_size=3,
            strides=1,
            dilation_rate=3,
            padding="same",
        )(x_2)

        # 分支 3
        x_3 = Conv2D(branch_channels, kernel_size=1, strides=1, padding="same")(input)
        x_3 = LeakyReLU(0.2)(x_3)
        x_3 = Conv2D(branch_channels, kernel_size=(3, 1), strides=1, padding="same")(
            x_3
        )
        x_3 = LeakyReLU(0.2)(x_3)
        x_3 = Conv2D(
            branch_channels,
            kernel_size=3,
            strides=1,
            dilation_rate=3,
            padding="same",
        )(x_3)

        # 分支 4
        x_4 = Conv2D(branch_channels // 2, kernel_size=1, strides=1, padding="same")(
            input
        )
        x_4 = LeakyReLU(0.2)(x_4)
        x_4 = Conv2D(
            (branch_channels // 4) * 3,
            kernel_size=(1, 3),
            strides=1,
            padding="same",
        )(x_4)
        x_4 = LeakyReLU(0.2)(x_4)
        x_4 = Conv2D(branch_channels, kernel_size=(1, 3), strides=1, padding="same")(
            x_4
        )
        x_4 = LeakyReLU(0.2)(x_4)
        x_4 = Conv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            dilation_rate=5,
            padding="same",
        )(x_4)

        x_4 = Concatenate()([x_1, x_2, x_3, x_4])
        x_4 = Conv2D(out_channels, kernel_size=1, strides=1, padding="same")(x_4)
        output = Add()([x_4, shortcut])

        return output

    def RFDB(input, in_channels=64, growth_channels=32):
        x_1 = RFB(
            input,
            in_channels=in_channels,
            out_channels=growth_channels,
        )
        x_1 = LeakyReLU(0.2)(x_1)
        x_1 = Concatenate()([input, x_1])

        x_2 = RFB(
            x_1,
            in_channels=in_channels + growth_channels,
            out_channels=growth_channels,
        )
        x_2 = LeakyReLU(0.2)(x_2)
        x_2 = Concatenate()([input, x_1, x_2])

        x_3 = RFB(
            x_2,
            in_channels=in_channels + growth_channels * 2,
            out_channels=growth_channels,
        )
        x_3 = LeakyReLU(0.2)(x_3)
        x_3 = Concatenate()([input, x_1, x_2, x_3])

        x_4 = RFB(
            x_3,
            in_channels=in_channels + growth_channels * 3,
            out_channels=growth_channels,
        )
        x_4 = LeakyReLU(0.2)(x_4)
        x_4 = Concatenate()([input, x_1, x_2, x_3, x_4])

        x_5 = RFB(
            x_4,
            in_channels=in_channels + growth_channels * 4,
            out_channels=in_channels,
        )
        x_5 = Lambda(lambda x: x * 0.2)(x_5)
        output = Add()([x_5, input])

        return output

    def RRFDB(input, input_channels=64, growth_channels=32):
        x = RFDB(input, input_channels, growth_channels)
        x = RFDB(x, input_channels, growth_channels)
        x = RFDB(x, input_channels, growth_channels)

        x = Lambda(lambda x: x * 0.2)(x)
        output = Add()([x, input])

        return output

    # 低分辨率图像作为输入
    lr_input = Input(shape=(32, 32, 3))

    # RRDB 之前
    x_start = Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
    )(lr_input)
    x_start = LeakyReLU(0.5)(x_start)

    # RRDB
    x = x_start
    for _ in range(16):
        x = RRDB(x)

    # RRFDB
    for _ in range(8):
        x = RRFDB(x)

    # RFB
    x = RFB(x, in_channels=64, out_channels=64)
    x = Add()([x, x_start])

    # 上采样
    for i in range(4 // 2):
        x = upsample(x, i + 1)  # 每次上采样，图像尺寸变为原来的两倍

    x = Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
    )(x)
    x = LeakyReLU(0.2)(x)
    hr_output = Conv2D(3, kernel_size=3, strides=1, padding="same", activation="tanh")(
        x
    )

    model = Model(inputs=lr_input, outputs=hr_output, name="generator")
    model.summary()

    return model


if __name__ == "__main__":
    build_generator()
