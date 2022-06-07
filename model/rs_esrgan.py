from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Model

from model.esrgan import ESRGAN


class RS_ESRGAN(ESRGAN):
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
        # 构建上采样模块
        def upsample(x, number, method="nni", channels=64):
            # 最近邻域插值上采样
            if method == "nni":
                x = UpSampling2D(
                    size=2,
                    interpolation="nearest",
                    name="up_sample_nni_" + str(number),
                )(x)
                x = RFB(x, in_channels=channels, out_channels=channels)
                x = LeakyReLU(0.2)(x)
            # 亚像素卷积上采样
            elif method == "spc":
                x = Conv2D(
                    channels * 4,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    name="up_sample_conv2d_" + str(number),
                    kernel_constraint=self.sn_layer,
                )(x)
                x = self.subpixel_conv2d("up_sample_spc_" + str(number), 2)(x)
                x = RFB(x, in_channels=channels, out_channels=channels)
                x = LeakyReLU(0.2)(x)
            else:
                raise ValueError("Unsupported upsample method!")

            return x

        # 构建 dense block
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
                kernel_constraint=self.sn_layer,
            )(input)
            shortcut = Lambda(lambda x: x * 0.2)(shortcut)

            # 分支 1
            x_1 = Conv2D(
                branch_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(input)
            x_1 = LeakyReLU(0.2)(x_1)
            x_1 = Conv2D(
                branch_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_1)

            # 分支 2
            x_2 = Conv2D(
                branch_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(input)
            x_2 = LeakyReLU(0.2)(x_2)
            x_2 = Conv2D(
                branch_channels,
                kernel_size=(1, 3),
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_2)
            x_2 = LeakyReLU(0.2)(x_2)
            x_2 = Conv2D(
                branch_channels,
                kernel_size=3,
                strides=1,
                dilation_rate=3,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_2)

            # 分支 3
            x_3 = Conv2D(
                branch_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(input)
            x_3 = LeakyReLU(0.2)(x_3)
            x_3 = Conv2D(
                branch_channels,
                kernel_size=(3, 1),
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_3)
            x_3 = LeakyReLU(0.2)(x_3)
            x_3 = Conv2D(
                branch_channels,
                kernel_size=3,
                strides=1,
                dilation_rate=3,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_3)

            # 分支 4
            x_4 = Conv2D(
                branch_channels // 2,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(input)
            x_4 = LeakyReLU(0.2)(x_4)
            x_4 = Conv2D(
                (branch_channels // 4) * 3,
                kernel_size=(1, 3),
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_4)
            x_4 = LeakyReLU(0.2)(x_4)
            x_4 = Conv2D(
                branch_channels,
                kernel_size=(1, 3),
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_4)
            x_4 = LeakyReLU(0.2)(x_4)
            x_4 = Conv2D(
                out_channels,
                kernel_size=3,
                strides=1,
                dilation_rate=5,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_4)

            x_4 = Concatenate()([x_1, x_2, x_3, x_4])
            x_4 = Conv2D(
                out_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_constraint=self.sn_layer,
            )(x_4)
            output = Add()([x_4, shortcut])

            return output

        # 构建 RFDB
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

        # 构建 RRFDB
        def RRFDB(input, input_channels=64, growth_channels=32):
            x = RFDB(input, input_channels, growth_channels)
            x = RFDB(x, input_channels, growth_channels)
            x = RFDB(x, input_channels, growth_channels)

            x = Lambda(lambda x: x * 0.2)(x)
            output = Add()([x, input])

            return output

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

        # RRFDB
        for _ in range(8):
            x = RRFDB(x)

        # RRFDB 之后
        x = RFB(x, in_channels=64, out_channels=64)
        x = Add()([x, x_start])

        # 交替使用最近邻域插值和亚像素卷积上采样方法
        for i in range(self.scale_factor // 2):
            # 每次上采样，图像尺寸变为原来的两倍
            if (i + 1) % 2 == 0:
                x = upsample(x, i + 1, method="spc", channels=64)
            else:
                x = upsample(x, i + 1, method="nni", channels=64)

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
