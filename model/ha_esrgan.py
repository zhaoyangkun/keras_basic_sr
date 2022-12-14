from tensorflow.keras.layers import Activation, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model

from model.real_esrgan import RealESRGAN
from util.layer import MHARG, upsample_mharb


class HA_ESRGAN(RealESRGAN):

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
        rdb_num=16,
        max_workers=4,
        data_enhancement_factor=1,
        log_interval=20,
        save_images_interval=10,
        save_models_interval=50,
        save_history_interval=10,
        pretrain_model_path="",
        use_mixed_float=False,
        use_sn=False,
        use_ema=False,
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
            rdb_num,
            max_workers,
            data_enhancement_factor,
            log_interval,
            save_images_interval,
            save_models_interval,
            save_history_interval,
            pretrain_model_path,
            use_mixed_float,
            use_sn,
            use_ema,
        )

    def build_generator(self):
        """
        构建生成器
        """
        # 低分辨率图像作为输入
        lr_input = Input(shape=(None, None, 3))

        x_start = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(lr_input)
        x_start = LeakyReLU(alpha=0.2)(x_start)

        x = x_start
        # 残差组
        for _ in range(8):
            x = MHARG(x)

        # 交替使用双线性插值和亚像素卷积上采样算法
        for i in range(self.scale_factor // 2):
            # 每次上采样，图像尺寸变为原来的两倍
            if (i + 1) % 2 == 0:
                x = upsample_mharb(x, i + 1, method="subpixel", channels=64)
            else:
                x = upsample_mharb(x, i + 1, method="bilinear", channels=64)

        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
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

    def content_loss(self, hr_img, hr_generated):
        return super().content_loss(hr_img, hr_generated) * 0.001
