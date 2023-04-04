import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    LeakyReLU,
)
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from util.generate import denormalize
from util.layer import RRDB, create_vgg_19_features_model, upsample

from model.srgan import SRGAN


class ESRGAN(SRGAN):
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

        # 优化器
        self.pre_optimizer = Adam(2e-4)
        self.gen_optimizer = Adam(1e-4)
        self.dis_optimizer = Adam(1e-4)

        # 检查是否使用混合精度
        self.check_mixed()

        # 损失权重
        self.loss_weights = {"percept": 1, "gen": 5e-3, "pixel": 1e-2}

        # 损失函数
        self.mse_loss = MeanSquaredError()
        self.mae_loss = MeanAbsoluteError()

    def build_vgg(self):
        """
        构建 vgg 模型
        """
        # vgg = VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        # vgg.layers[20].activation = None

        # model = Model(vgg.input, vgg.layers[20].output)
        # model.trainable = False

        # return model
        return create_vgg_19_features_model(loss_type="esrgan")

    def build_generator(self):
        """
        构建生成器
        """
        # 低分辨率图像作为输入
        lr_img = Input(shape=(None, None, 3))

        # RRDB 之前
        x_start = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(lr_img)
        x_start = LeakyReLU(0.5)(x_start)

        # RRDB
        x = x_start
        for _ in range(8):  # 默认为 16 个
            x = RRDB(x)

        # RRDB 之后
        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
        )(x)
        x = Add()([x, x_start])

        # 上采样
        for i in range(self.scale_factor // 2):
            x = upsample(x, i + 1, 64)  # 每次上采样，图像尺寸变为原来的两倍

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
        sr_img = Activation(
            "tanh",
            dtype="float32",
        )(x)

        model = Model(inputs=lr_img, outputs=sr_img, name="generator")
        # model.summary()

        return model

    def build_discriminator(self, filters=64):
        """构建判别器

        Args:
            filters (int, optional): 通道数. 默认为 64.
        """
        # 基本卷积块
        def conv2d_block(input, filters, strides=1, bn=True):
            x = Conv2D(filters, 3, strides=strides, padding="same")(input)
            if bn:
                x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            return x

        # 高分辨率图像作为输入
        img = Input(shape=self.hr_shape)  # (h, w, 3)
        x = conv2d_block(
            img,
            filters,
            bn=False,
        )  # (h, w, filters)
        x = conv2d_block(
            x,
            filters,
            strides=2,
        )  # (h/2, w/2, filters)
        x = conv2d_block(
            x,
            filters * 2,
        )  # (h/2, w/2, filters * 2)
        x = conv2d_block(
            x,
            filters * 2,
            strides=2,
        )  # (h/4, w/4, filters * 2)
        x = conv2d_block(
            x,
            filters * 4,
        )  # (h/4, w/4, filters * 4)
        x = conv2d_block(
            x,
            filters * 4,
            strides=2,
        )  # (h/8, w/8, filters * 4)
        x = conv2d_block(
            x,
            filters * 8,
        )  # (h/8, w/8, filters * 8)
        x = conv2d_block(
            x,
            filters * 8,
            strides=2,
        )  # (h/16, w/16, filters * 8)
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)  # (filters * 8)
        x = Dense(filters * 16)(x)  # (filters * 16)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1, dtype="float32")(x)  # (1)

        model = Model(inputs=img, outputs=x, name="discriminator")
        # model.summary()

        return model

    def content_loss(self, hr_img, hr_generated):
        """
        内容损失
        """
        # 反归一化 vgg 输入
        hr_generated = denormalize(hr_generated, (-1, 1))
        hr_img = denormalize(hr_img, (-1, 1))

        hr_generated_features = self.vgg(hr_generated) / 12.75
        hr_features = self.vgg(hr_img) / 12.75

        return self.mse_loss(hr_features, hr_generated_features)

    def generator_loss(self, real_logit, fake_logit):
        """
        生成器损失
        """
        generator_loss = K.mean(
            K.binary_crossentropy(K.zeros_like(real_logit), real_logit)
            + K.binary_crossentropy(K.ones_like(fake_logit), fake_logit)
        )

        return generator_loss

    def discriminator_loss(self, fake_logit, real_logit):
        """
        判别器损失
        """
        discriminator_loss = K.mean(
            K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit)
            + K.binary_crossentropy(K.ones_like(real_logit), real_logit)
        )

        return discriminator_loss

    def relativistic_loss(self, real_img, fake_img):
        """
        相对损失
        """
        fake_logit = K.sigmoid(fake_img - K.mean(real_img))
        real_logit = K.sigmoid(real_img - K.mean(fake_img))

        return fake_logit, real_logit

    def pretrain_scheduler(self, optimizers, mini_batches):
        """
        预训练中动态修改学习率
        """
        # 每隔 200000 次 minibatch，学习率衰减一半
        if mini_batches % (200000) == 0:
            for optimizer in optimizers:
                lr = K.get_value(optimizer.lr)
                K.set_value(optimizer.lr, lr * 0.5)
                self.logger.info("pretrain lr changed to {}".format(lr * 0.5))

    def scheduler(self, optimizers, epoch):
        """
        动态修改学习率
        """
        if epoch in [50000, 100000, 200000, 300000]:
            for optimizer in optimizers:
                lr = K.get_value(optimizer.lr)
                K.set_value(optimizer.lr, lr * 0.5)
                self.logger.info("train lr changed to {}".format(lr * 0.5))

    @tf.function
    def pretrain_step(self, lr_img, hr_img):
        """
        单步预训练
        """
        with tf.GradientTape() as tape:
            hr_generated = self.generator(lr_img, training=True)
            loss = self.mae_loss(hr_img, hr_generated)

            if self.use_mixed_float:
                scaled_loss = self.pre_optimizer.get_scaled_loss(loss)
        if self.use_mixed_float:
            scaled_gradients = tape.gradient(
                scaled_loss, self.generator.trainable_variables
            )
            gradients = self.pre_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.generator.trainable_variables)

        self.pre_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return loss

    @tf.function
    def train_step(self, lr_img, hr_img):
        """
        单步训练
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            hr_generated = self.generator(lr_img, training=True)

            # ***
            # 判别器
            # ***
            hr_output = self.discriminator(hr_img, training=True)
            hr_generated_output = self.discriminator(hr_generated, training=True)

            fake_logit, real_logit = self.relativistic_loss(
                hr_output, hr_generated_output
            )

            discriminator_loss = self.discriminator_loss(fake_logit, real_logit)

            # ***
            # 生成器
            # ***
            percept_loss = self.content_loss(hr_img, hr_generated)
            generator_loss = self.generator_loss(real_logit, fake_logit)
            pixel_loss = self.mae_loss(hr_img, hr_generated)
            generator_total_loss = (
                self.loss_weights["percept"] * percept_loss
                + self.loss_weights["gen"] * generator_loss
                + self.loss_weights["pixel"] * pixel_loss
            )

            # 若使用混合精度
            if self.use_mixed_float:
                # 将损失值乘以损失标度值
                scaled_disc_loss = self.dis_optimizer.get_scaled_loss(
                    discriminator_loss
                )
                scaled_gen_loss = self.gen_optimizer.get_scaled_loss(
                    generator_total_loss
                )

                # # 将数值类型从 float16 转换为 float32
                # hr_generated = tf.cast(hr_generated, dtype=tf.float32)
                # generator_total_loss = tf.cast(generator_total_loss, dtype=tf.float32)
                # discriminator_loss = tf.cast(discriminator_loss, dtype=tf.float32)

            # # 将归一化区间从 [-1, 1] 转换到 [0, 255]
            # hr_img = tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8)
            # hr_generated = tf.cast((hr_generated + 1) * 127.5, dtype=tf.uint8)

            # 计算 psnr，ssim
            # psnr = calculate_psnr(
            #     hr_img,
            #     hr_generated,
            #     crop_border=0,
            #     input_order="HWC",
            #     test_y_channel=True,
            # )
            # ssim = calculate_ssim(
            #     hr_img,
            #     hr_generated,
            #     crop_border=0,
            #     input_order="HWC",
            #     test_y_channel=True,
            # )
            # psnr = tf.reduce_mean(tf.image.psnr(hr_img, hr_generated, max_val=1.0))
            # ssim = tf.reduce_mean(tf.image.ssim(hr_img, hr_generated, max_val=1.0))

        # 若使用混合精度，将梯度除以损失标度
        if self.use_mixed_float:
            scaled_gen_gradients = gen_tape.gradient(
                scaled_gen_loss, self.generator.trainable_variables
            )
            scaled_dis_gradients = disc_tape.gradient(
                scaled_disc_loss, self.discriminator.trainable_variables
            )
            gradients_generator = self.gen_optimizer.get_unscaled_gradients(
                scaled_gen_gradients
            )
            gradients_discriminator = self.dis_optimizer.get_unscaled_gradients(
                scaled_dis_gradients
            )
        # 不使用混合精度，直接获取梯度
        else:
            gradients_generator = gen_tape.gradient(
                generator_total_loss, self.generator.trainable_variables
            )
            gradients_discriminator = disc_tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables
            )
        # 更新优化器参数
        self.gen_optimizer.apply_gradients(
            zip(gradients_generator, self.generator.trainable_variables)
        )
        self.dis_optimizer.apply_gradients(
            zip(gradients_discriminator, self.discriminator.trainable_variables)
        )

        return generator_total_loss, discriminator_loss
