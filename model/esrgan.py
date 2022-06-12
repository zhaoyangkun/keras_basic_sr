import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    LeakyReLU,
    PReLU,
)
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from util.data_loader import DataLoader
from util.layer import spectral_norm_conv2d
from util.logger import create_logger


class ESRGAN:
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
        self.model_name = model_name  # 模型名称
        self.result_path = result_path  # 结果保存路径
        self.train_resource_path = train_resource_path  # 训练图片资源路径
        self.test_resource_path = test_resource_path  # 测试图片资源路径
        self.epochs = epochs  # 训练轮数
        self.init_epoch = init_epoch  # 初始化训练轮数
        self.batch_size = batch_size  # 批次大小
        self.downsample_mode = downsample_mode  # 下采样模式
        self.scale_factor = scale_factor  # 图片缩放比例
        self.lr_shape = (
            train_hr_img_height // scale_factor,
            train_hr_img_width // scale_factor,
            3,
        )  # 缩放后的图片尺寸
        self.hr_shape = (train_hr_img_height, train_hr_img_width, 3)  # 原图尺寸
        self.train_hr_img_height = train_hr_img_height
        self.train_hr_img_width = train_hr_img_width
        self.valid_hr_img_height = valid_hr_img_height
        self.valid_hr_img_width = valid_hr_img_width
        self.max_workers = max_workers  # 处理图片的最大线程数
        self.data_enhancement_factor = (
            data_enhancement_factor  # 数据增强因子，表示利用随机裁剪和水平翻转来扩充训练数据集的倍数，默认为 1（不进行扩充）
        )
        self.log_interval = log_interval  # 打印日志间隔
        self.save_images_interval = save_images_interval  # 保存图片迭代间隔
        self.save_models_interval = save_models_interval  # 保存模型迭代间隔
        self.save_history_interval = save_history_interval  # 保存历史数据迭代间隔
        self.pretrain_model_path = pretrain_model_path  # 预训练模型路径

        # 创建日志记录器
        log_dir_path = os.path.join(self.result_path, self.model_name, "logs")
        log_file_name = "%s_train.log" % self.model_name
        self.logger = create_logger(log_dir_path, log_file_name, self.model_name)

        # 创建数据集
        self.data_loader = DataLoader(
            self.train_resource_path,
            self.test_resource_path,
            self.batch_size,
            self.downsample_mode,
            self.train_hr_img_height,
            self.train_hr_img_width,
            self.valid_hr_img_height,
            self.valid_hr_img_width,
            self.scale_factor,
            self.max_workers,
            self.data_enhancement_factor,
        )

        # 优化器
        self.pre_optimizer = Adam(2e-4)
        self.gen_optimizer = Adam(1e-4)
        self.dis_optimizer = Adam(1e-4)

        # 损失权重
        self.loss_weights = {"percept": 1, "gen": 5e-3, "pixel": 1e-2}

        # 损失函数
        self.mse_loss = MeanSquaredError()
        self.mae_loss = MeanAbsoluteError()

        # 谱归一化
        self.use_sn = use_sn

        # 构建 vgg 模型
        self.vgg = self.build_vgg()

        # 构建生成器
        self.generator = self.build_generator()

        # 构建判别器
        self.discriminator = self.build_discriminator()

    def subpixel_conv2d(self, name, scale=2):
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

    def build_vgg(self):
        """
        构建 vgg 模型
        """
        vgg = VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)

        model = Model(vgg.input, outputs=vgg.layers[20].output)
        model.trainable = False

        return model

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
            )(input)
            x1 = LeakyReLU(0.2)(x1)

            x2 = Concatenate()([input, x1])
            x2 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
            )(x2)
            x2 = LeakyReLU(0.2)(x2)

            x3 = Concatenate()([input, x1, x2])
            x3 = Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
            )(x3)
            x3 = LeakyReLU(0.2)(x3)

            x4 = Concatenate()([input, x1, x2, x3])
            x4 = Conv2D(
                64,
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
        )(lr_input)
        x_start = LeakyReLU(0.5)(x_start)

        # RRDB
        x = x_start
        for _ in range(6):  # 默认为 16 个
            x = RRDB(x)

        # RRDB 之后
        x = Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
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
        )(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(
            3, kernel_size=3, strides=1, padding="same", activation="tanh"
        )(x)

        model = Model(inputs=lr_input, outputs=hr_output, name="generator")
        # model.summary()

        return model

    def build_discriminator(self, filters=64):
        """构建判别器

        Args:
            filters (int, optional): 通道数. 默认为 64.
        """
        # 基本卷积块
        def conv2d_block(input, filters, strides=1, bn=True, use_sn=True):
            x = spectral_norm_conv2d(
                input,
                use_sn,
                filters=filters,
                kernel_size=3,
                strides=strides,
                padding="same",
            )
            if bn:
                x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            return x

        # 高分辨率图像作为输入
        img = Input(shape=self.hr_shape)  # (h, w, 3)
        x = conv2d_block(img, filters, bn=False, use_sn=self.use_sn)  # (h, w, filters)
        x = conv2d_block(
            x, filters, strides=2, use_sn=self.use_sn
        )  # (h/2, w/2, filters)
        x = conv2d_block(x, filters * 2, use_sn=self.use_sn)  # (h/2, w/2, filters * 2)
        x = conv2d_block(
            x, filters * 2, strides=2, use_sn=self.use_sn
        )  # (h/4, w/4, filters * 2)
        x = conv2d_block(x, filters * 4, use_sn=self.use_sn)  # (h/4, w/4, filters * 4)
        x = conv2d_block(
            x, filters * 4, strides=2, use_sn=self.use_sn
        )  # (h/8, w/8, filters * 4)
        x = conv2d_block(x, filters * 8, use_sn=self.use_sn)  # (h/8, w/8, filters * 8)
        x = conv2d_block(
            x, filters * 8, strides=2, use_sn=self.use_sn
        )  # (h/16, w/16, filters * 8)
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)  # (filters * 8)
        x = Dense(filters * 16)(x)  # (filters * 16)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)  # (1)

        model = Model(inputs=img, outputs=x, name="discriminator")
        model.summary()

        return model

    def content_loss(self, hr_img, hr_generated):
        """
        内容损失
        """
        # 反归一化 vgg 输入
        def preprocess_vgg(x):
            if isinstance(x, np.ndarray):
                return preprocess_input((x + 1) * 127.5)
            else:
                return Lambda(lambda x: preprocess_input((x + 1) * 127.5))(x)

        hr_generated = preprocess_vgg(hr_generated)
        hr_img = preprocess_vgg(hr_img)

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

        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.pre_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return loss

    def pretrain(self):
        """
        预训练
        """
        # 保存历史数据文件夹路径
        save_history_dir_path = os.path.join(
            self.result_path, self.model_name, "history", "pretrain"
        )
        # 若保存历史数据文件夹不存在，则创建
        if not os.path.isdir(save_history_dir_path):
            os.makedirs(save_history_dir_path)

        # 保存模型文件夹路径
        save_models_dir_path = os.path.join(
            self.result_path, self.model_name, "models", "pretrain"
        )
        # 若保存模型文件夹不存在，则创建
        if not os.path.isdir(save_models_dir_path):
            os.makedirs(save_models_dir_path)

        epoch_list = tf.constant([])
        loss_list = tf.constant([], dtype=tf.float32)
        batch_idx_count = tf.constant(
            len(self.data_loader.train_data), dtype=tf.float32
        )
        for epoch in range(self.init_epoch, self.epochs + 1):
            loss_batch_total = tf.constant(0, dtype=tf.float32)

            # 加载训练数据集，并训练
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
                # 计算 mini-batch 数目
                mini_batches = (
                    (epoch - 1) * len(self.data_loader.train_data) + batch_idx + 1
                )

                # 根据 mini-batch 数目动态修改学习率
                self.pretrain_scheduler([self.pre_optimizer], mini_batches)

                # 单步训练
                loss = self.pretrain_step(lr_imgs, hr_imgs)
                # loss = loss.numpy().item()
                loss_batch_total += loss

                # 输出日志
                if (batch_idx + 1) % self.log_interval == 0:
                    self.logger.info(
                        "mode: pretrain, epochs: [%d/%d], batches: [%d/%d], loss: %.4f"
                        % (
                            epoch,
                            self.epochs,
                            batch_idx + 1,
                            batch_idx_count,
                            loss,
                        )
                    )

            # 统计 epoch 和 loss
            epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
            loss_list = tf.concat(
                [loss_list, [loss_batch_total / batch_idx_count]], axis=0
            )

            # 保存历史数据
            if epoch % self.save_history_interval == 0:
                self.save_pretrain_history(
                    epoch, save_history_dir_path, epoch_list, loss_list
                )

            # 保存模型
            if epoch % self.save_models_interval == 0:
                self.save_pretrain_models(epoch, save_models_dir_path)

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

            # 将归一化区间从 [-1, 1] 转换到 [0, 1]
            hr_img = (hr_img + 1) / 2
            hr_generated = (hr_generated + 1) / 2
            # 计算 psnr 和 ssim
            psnr = tf.reduce_mean(tf.image.psnr(hr_img, hr_generated, max_val=1.0))
            ssim = tf.reduce_mean(tf.image.ssim(hr_img, hr_generated, max_val=1.0))
        # 计算梯度
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

        return generator_total_loss, discriminator_loss, psnr, ssim

    def train(self):
        """
        训练模型
        """
        # 保存模型文件夹路径
        save_models_dir_path = os.path.join(
            self.result_path, self.model_name, "models", "train"
        )
        # 若保存模型文件夹不存在，则创建
        if not os.path.isdir(save_models_dir_path):
            os.makedirs(save_models_dir_path)

        # 保存图片文件夹路径
        save_images_dir_path = os.path.join(
            self.result_path, self.model_name, "images", "train"
        )
        # 若保存图片文件夹不存在，则创建
        if not os.path.isdir(save_images_dir_path):
            os.makedirs(save_images_dir_path)

        # 保存历史数据文件夹路径
        save_history_dir_path = os.path.join(
            self.result_path, self.model_name, "history", "train"
        )
        # 若保存历史数据文件夹不存在，则创建
        if not os.path.isdir(save_history_dir_path):
            os.makedirs(save_history_dir_path)

        # 加载预训练模型
        if self.pretrain_model_path:
            self.generator.load_weights(self.pretrain_model_path)

        # 若初始 epoch 大于 1，则加载模型
        if self.init_epoch > 1:
            self.generator = tf.keras.models.load_model(
                os.path.join(
                    save_models_dir_path, "gen_model_epoch_%d" % (self.init_epoch)
                )
            )
            self.discriminator = tf.keras.models.load_model(
                os.path.join(
                    save_models_dir_path, "dis_model_epoch_%d" % (self.init_epoch)
                )
            )

        epoch_list = tf.constant([])
        g_loss_list = tf.constant([], dtype=tf.float32)
        d_loss_list = tf.constant([], dtype=tf.float32)
        psnr_list = tf.constant([], dtype=tf.float32)
        ssim_list = tf.constant([], dtype=tf.float32)
        for epoch in range(self.init_epoch, self.epochs + 1):
            g_loss_batch_total = tf.constant(0, dtype=tf.float32)
            d_loss_batch_total = tf.constant(0, dtype=tf.float32)
            psnr_batch_total = tf.constant(0, dtype=tf.float32)
            ssim_batch_total = tf.constant(0, dtype=tf.float32)
            batch_idx_count = tf.constant(
                len(self.data_loader.train_data), dtype=tf.float32
            )

            # 修改学习率
            self.scheduler([self.gen_optimizer, self.dis_optimizer], epoch)

            # 加载训练数据集，并训练
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
                g_loss, d_loss, psnr, ssim = self.train_step(lr_imgs, hr_imgs)
                # g_loss, d_loss, psnr, ssim = (
                #     g_loss.numpy().item(),
                #     d_loss.numpy().item(),
                #     psnr.numpy().item(),
                #     ssim.numpy().item(),
                # )

                g_loss_batch_total += g_loss
                d_loss_batch_total += d_loss
                psnr_batch_total += psnr
                ssim_batch_total += ssim

                # 输出日志
                if (batch_idx + 1) % self.log_interval == 0:
                    self.logger.info(
                        "mode: train, epochs: [%d/%d], batches: [%d/%d], d_loss: %.4f, g_loss: %.4f, psnr: %.2f, ssim: %.2f"
                        % (
                            epoch,
                            self.epochs,
                            batch_idx + 1,
                            batch_idx_count,
                            d_loss,
                            g_loss,
                            psnr,
                            ssim,
                        )
                    )

            epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
            g_loss_list = tf.concat(
                [g_loss_list, [g_loss_batch_total / batch_idx_count]], axis=0
            )
            d_loss_list = tf.concat(
                [d_loss_list, [d_loss_batch_total / batch_idx_count]], axis=0
            )
            psnr_list = tf.concat(
                [psnr_list, [psnr_batch_total / batch_idx_count]], axis=0
            )
            ssim_list = tf.concat(
                [ssim_list, [ssim_batch_total / batch_idx_count]], axis=0
            )

            # 保存历史数据
            if epoch % self.save_history_interval == 0:
                self.save_history(
                    epoch,
                    save_history_dir_path,
                    epoch_list,
                    g_loss_list,
                    d_loss_list,
                    psnr_list,
                    ssim_list,
                )

            # 保存图片
            if epoch % self.save_images_interval == 0:
                self.save_images(epoch, save_images_dir_path, 5)

            # 评估并保存模型
            if epoch % self.save_models_interval == 0:
                self.evaluate(epoch)
                self.save_models(epoch, save_models_dir_path)

    @tf.function
    def valid_step(self, lr_img, hr_img):
        """
        单步验证
        """
        hr_generated = self.generator(lr_img, training=False)

        # 将归一化区间从 [-1, 1] 转换到 [0, 1]
        hr_img = (hr_img + 1) / 2
        hr_generated = (hr_generated + 1) / 2

        # 计算 psnr 和 ssim
        psnr = tf.reduce_mean(tf.image.psnr(hr_img, hr_generated, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(hr_img, hr_generated, max_val=1.0))

        return psnr, ssim

    def evaluate(self, epoch):
        """
        评估模型
        """
        test_data_len = tf.constant(len(self.data_loader.test_data), dtype=tf.float32)
        psnr_total = tf.constant(0, dtype=tf.float32)
        ssim_total = tf.constant(0, dtype=tf.float32)
        for (lr_imgs, hr_imgs) in self.data_loader.test_data:
            psnr, ssim = self.valid_step(lr_imgs, hr_imgs)
            # 统计 PSNR 和 SSIM
            psnr_total += psnr
            ssim_total += ssim
        # 输出 SSIM 和 PSNR
        self.logger.info(
            "evaluate %s, epochs: [%d/%d], PSNR: %.2f, SSIM: %.2f"
            % (
                self.model_name,
                epoch,
                self.epochs,
                psnr_total / test_data_len,
                ssim_total / test_data_len,
            )
        )

    def save_pretrain_history(
        self, epoch, save_history_dir_path, epoch_list, loss_list
    ):
        """
        保存预训练历史数据
        """
        fig = plt.figure(figsize=(10, 10))

        # 绘制损失曲线
        ax = plt.subplot(1, 1, 1)
        ax.set_title("Pretrain Loss")
        (line,) = ax.plot(
            epoch_list, loss_list, color="deepskyblue", marker=".", label="loss"
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        ax.legend(handles=[line], loc="upper right")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                save_history_dir_path, "pretrain_history_epoch_%d.png" % epoch
            ),
            dpi=500,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

    def save_history(
        self,
        epoch,
        save_history_dir_path,
        epoch_list,
        g_loss_list,
        d_loss_list,
        psnr_list,
        ssim_list,
    ):
        """
        保存历史数据
        """
        fig = plt.figure(figsize=(10, 10))

        # 绘制损失曲线
        ax_1 = plt.subplot(2, 1, 1)
        ax_1.set_title("Train Loss")
        (line_1,) = ax_1.plot(
            epoch_list, g_loss_list, color="deepskyblue", marker=".", label="g_loss"
        )
        (line_2,) = ax_1.plot(
            epoch_list, d_loss_list, color="darksalmon", marker=".", label="d_loss"
        )
        ax_1.set_xlabel("epoch")
        ax_1.set_ylabel("Loss")
        ax_1.legend(handles=[line_1, line_2], loc="upper right")

        # 绘制 PSNR 曲线
        ax_2 = plt.subplot(2, 2, 3)
        ax_2.set_title("PSNR")
        (line_3,) = ax_2.plot(
            epoch_list, psnr_list, color="orange", marker=".", label="PSNR"
        )
        ax_2.set_xlabel("epoch")
        ax_2.set_ylabel("PSNR")
        ax_2.legend(handles=[line_3], loc="upper left")

        # 绘制损失曲线
        ax_3 = plt.subplot(2, 2, 4)
        ax_3.set_title("SSIM")
        (line_4,) = ax_3.plot(
            epoch_list, ssim_list, color="skyblue", marker=".", label="SSIM"
        )
        ax_3.set_xlabel("epoch")
        ax_3.set_ylabel("SSIM")
        ax_3.legend(handles=[line_4], loc="upper left")

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_history_dir_path, "train_history_epoch_%d.png" % epoch),
            dpi=500,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

    def save_images(self, epoch, save_images_dir_path, take_num=5):
        """
        保存图片
        """
        # 从测试数据集中取出一批图片
        test_dataset = self.data_loader.test_data.unbatch().take(take_num)

        # 绘图
        fig, axs = plt.subplots(take_num, 3)
        for i, (lr_img, hr_img) in enumerate(test_dataset):
            # 利用生成器生成图片
            sr_img = self.generator.predict(tf.expand_dims(lr_img, 0))
            sr_img = tf.squeeze(sr_img, axis=0)

            # 反归一化
            lr_img, hr_img, sr_img = (
                tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8),
                tf.cast((hr_img + 1) * 127.5, dtype=tf.uint8),
                tf.cast((sr_img + 1) * 127.5, dtype=tf.uint8),
            )

            axs[i, 0].imshow(lr_img)
            axs[i, 0].axis("off")
            if i == 0:
                axs[i, 0].set_title("Bicubic")

            axs[i, 1].imshow(sr_img)
            axs[i, 1].axis("off")
            if i == 0:
                axs[i, 1].set_title("ESRGAN")

            axs[i, 2].imshow(hr_img)
            axs[i, 2].axis("off")
            if i == 0:
                axs[i, 2].set_title("Ground Truth")
        # 保存图片
        fig.savefig(
            os.path.join(save_images_dir_path, "test_epoch_%d.png" % epoch),
            dpi=500,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

    def save_pretrain_models(self, epoch, save_models_dir_path):
        """
        保存预训练模型
        """
        # 删除原先的模型文件，仅保存最新的模型
        file_name_list = os.listdir(save_models_dir_path)
        for file_name in file_name_list:
            file_path = os.path.join(save_models_dir_path, file_name)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)

        # 保存生成器权重
        self.generator.save_weights(
            os.path.join(save_models_dir_path, "gen_weights_epoch_%d.ckpt" % epoch),
            save_format="tf",
        )

    def save_models(self, epoch, save_models_dir_path):
        """
        保存模型
        """
        # 删除原先的模型文件，仅保存最新的模型
        file_name_list = os.listdir(save_models_dir_path)
        for file_name in file_name_list:
            file_path = os.path.join(save_models_dir_path, file_name)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)

        # 保存生成器
        self.generator.save(
            os.path.join(save_models_dir_path, "gen_model_epoch_%d" % epoch),
            save_format="tf",
        )

        # 保存判别器
        self.discriminator.save(
            os.path.join(save_models_dir_path, "dis_model_epoch_%d" % epoch),
            save_format="tf",
        )
