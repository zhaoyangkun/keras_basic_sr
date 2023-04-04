import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Add, Input, Lambda, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from util.generate import denormalize
from util.layer import create_vgg_19_features_model, spectral_norm_conv2d

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
        self.loss_weights = {"percept": 1, "gen": 0.1, "pixel": 1}

    def build_discriminator(self, filters=64):
        """
        构建判别器
        """
        input = Input(shape=self.hr_shape)  # (h, w, 3)

        # 第一层卷积
        x_0 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(
            input
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
        out = Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            dtype="float32",
        )(
            out
        )  # (h, w, 1)

        model = Model(inputs=input, outputs=out, name="discriminator")
        # model.summary()

        return model

    def build_vgg(self):
        """
        构建 vgg 模型
        """
        return create_vgg_19_features_model(loss_type="real-esrgan")

    def content_loss(self, hr_img, hr_generated):
        """
        内容损失
        """
        # 反归一化
        hr_generated = denormalize(hr_generated, (-1, 1))
        hr_img = denormalize(hr_img, (-1, 1))

        # 计算 vgg 特征
        hr_generated_features = self.vgg(hr_generated)
        hr_features = self.vgg(hr_img)

        loss_weights = tf.constant([0.1, 0.1, 1, 1, 1], dtype=tf.float32)
        content_loss = tf.constant(0.0, dtype=tf.float32)
        # 根据权重比计算内容损失
        for i in range(len(loss_weights)):
            content_loss += loss_weights[i] * self.mae_loss(
                hr_features[i], hr_generated_features[i]
            )

        return content_loss
