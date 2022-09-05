from tensorflow.keras.layers import Activation, Add, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model
from util.layer import RFB, RRDB, RRFDB, upsample_rfb

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
        downsample_mode="bicubic",
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
        use_sn=False,
        use_mixed_float=False,
    ):
        super().__init__(
            model_name,
            result_path,
            train_resource_path,
            test_resource_path,
            epochs,
            init_epoch,
            batch_size,
            downsample_mode,
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
            use_mixed_float,
        )

    def build_generator(self):
        """
        构建生成器
        """
        # 低分辨率图像作为输入
        lr_input = Input(shape=(None, None, 3))

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
        for _ in range(4):  # 默认为 16 块
            x = RRDB(x)

        # RRFDB
        for _ in range(2):  # 默认为 8 块
            x = RRFDB(x)

        # RRFDB 之后
        x = RFB(x, in_channels=64, out_channels=64)
        x = Add()([x, x_start])

        # 交替使用最近领域插值和亚像素卷积上采样算法
        for i in range(self.scale_factor // 2):
            # 每次上采样，图像尺寸变为原来的两倍
            if (i + 1) % 2 == 0:
                x = upsample_rfb(x, i + 1, method="subpixel", channels=64)
            else:
                x = upsample_rfb(x, i + 1, method="nearest", channels=64)

        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(
            3,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x)
        hr_output = Activation(
            "tanh",
            dtype="float32",
        )(x)

        model = Model(inputs=lr_input, outputs=hr_output, name="generator")
        # model.summary()

        return model
