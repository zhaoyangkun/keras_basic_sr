from tensorflow.keras.layers import Add, Input, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Model
from util.layer import spectral_norm_conv2d

from model.esrgan import ESRGAN


class RealESRGAN(ESRGAN):
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
        )

    def build_discriminator(self, filters=64):
        """
        构建判别器
        """
        input = Input(shape=self.hr_shape)  # (h, w, 3)

        # 第一层卷积
        x_0 = spectral_norm_conv2d(
            input,
            self.use_sn,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
        )  # (h, w, filters)
        x_0 = LeakyReLU(0.2)(x_0)

        # 第二层卷积
        x_1 = spectral_norm_conv2d(
            x_0,
            self.use_sn,
            filters=filters * 2,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )  # (h / 2, w / 2, filters * 2)
        x_1 = LeakyReLU(0.2)(x_1)

        # 第三层卷积
        x_2 = spectral_norm_conv2d(
            x_1,
            self.use_sn,
            filters=filters * 4,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )  # (h / 4, w / 4, filters * 4)
        x_2 = LeakyReLU(0.2)(x_2)

        # 第四层卷积
        x_3 = spectral_norm_conv2d(
            x_2,
            self.use_sn,
            filters=filters * 8,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )  # (h / 8, w / 8, filters * 8)
        x_3 = LeakyReLU(0.2)(x_3)

        # 上采样
        x_3 = UpSampling2D(interpolation="bilinear")(x_3)  # (h /4, h / 4, filters * 8)

        # 第五层卷积
        x_4 = spectral_norm_conv2d(
            x_3,
            self.use_sn,
            filters=filters * 4,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h / 4, w / 4, filters * 4)
        x_4 = LeakyReLU(0.2)(x_4)
        # 跳跃连接
        x_4 = Add()([x_4, x_2])

        # 上采样
        x_4 = UpSampling2D(interpolation="bilinear")(x_4)  # (h / 2, w / 2, filters * 4)

        # 第六层卷积
        x_5 = spectral_norm_conv2d(
            x_4,
            self.use_sn,
            filters=filters * 2,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h / 2, w / 2, filters * 2)
        x_5 = LeakyReLU(0.2)(x_5)
        # 跳跃连接
        x_5 = Add()([x_5, x_1])

        # 上采样
        x_5 = UpSampling2D(interpolation="bilinear")(x_5)  # (h, w, filters * 2)

        # 第七层卷积
        x_6 = spectral_norm_conv2d(
            x_5,
            self.use_sn,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h, w, filters)
        x_6 = LeakyReLU(0.2)(x_6)
        # 跳跃连接
        x_6 = Add()([x_6, x_0])

        # 第八层卷积
        out = spectral_norm_conv2d(
            x_6,
            self.use_sn,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h, w, filters)
        out = LeakyReLU(0.2)(out)

        # 第九层卷积
        out = spectral_norm_conv2d(
            out,
            self.use_sn,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h, w, filters)
        out = LeakyReLU(0.2)(out)

        # 第十层卷积
        out = spectral_norm_conv2d(
            out,
            self.use_sn,
            filters=1,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )  # (h, w, 1)

        model = Model(inputs=input, outputs=out, name="discriminator")
        model.summary()

        return model
