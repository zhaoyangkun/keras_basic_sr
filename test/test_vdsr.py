from tensorflow.keras.layers import Add, Conv2D, Input, ReLU
from tensorflow.keras.models import Model


def build_generator():
    """
    构建生成器
    """

    def conv2d_relu(x, filters, kernel_size, strides=1, padding="same"):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = ReLU()(x)
        return x

    inputs = Input(shape=[None, None, 1])

    x = conv2d_relu(inputs, 64, 3)

    for _ in range(18):
        x = conv2d_relu(x, 64, 3)

    x = Conv2D(3, 3, padding="same")(x)

    outputs = Add()([inputs, x])

    return Model(inputs=inputs, outputs=outputs)


genertor = build_generator()
genertor.summary()
