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

from esrgan import ESRGAN


class RESRGAN(ESRGAN):
    def __init__(
        self,
        model_name,
        result_path,
        train_resource_path,
        test_resource_path,
        epochs,
        init_epoch=1,
        batch_size=4,
        scale_factor=4,
        train_hr_img_height=128,
        train_hr_img_width=128,
        valid_hr_img_height=128,
        valid_hr_img_width=128,
        max_workers=4,
        data_enhancement_factor=1,
        log_interval=20,
        save_images_interval=10,
        save_models_interval=50,
        save_history_interval=10,
        pretrain_model_path="",
        use_sn=True,
    ):
        super().__init__(
            model_name,
            result_path,
            train_resource_path,
            test_resource_path,
            epochs,
            init_epoch,
            batch_size,
            scale_factor,
            train_hr_img_height,
            train_hr_img_width,
            valid_hr_img_height,
            valid_hr_img_width,
            max_workers,
            data_enhancement_factor,
            log_interval,
            save_images_interval,
            save_models_interval,
            save_history_interval,
            pretrain_model_path,
            use_sn,
        )

    def build_generator(self):
        """
        构建生成器
        """

        def dense_block(input):
            x1 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])

            x5 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
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

        def upsample(x, number):
            x = Conv2D(
                256,
                kernel_size=3,
                strides=1,
                padding="same",
                name="up_sample_conv2d_" + str(number),
            )(x)
            x = self.subpixel_conv2d("up_sample_subpixel_" + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name="up_sample_prelu_" + str(number))(x)

            return x

        # 低分辨率图像作为输入
        lr_input = Input(shape=(None, None, 3))

        # RRDB 之前
        x_start = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_constraint=self.sn_layer,
        )(lr_input)
        x_start = LeakyReLU(0.5)(x_start)

        # RRDB
        x = x_start
        for _ in range(16):
            x = RRDB(x)

        # RRDB 之后
        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_constraint=self.sn_layer,
        )(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # 上采样
        for i in range(self.scale_factor // 2):
            x = upsample(x, i + 1)  # 每次上采样，图像尺寸变为原来的两倍

        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_constraint=self.sn_layer,
        )(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(
            3, kernel_size=3, strides=1, padding="same", activation="tanh"
        )(x)

        model = Model(inputs=lr_input, outputs=hr_output, name="generator")
        model.summary()

        return model
