import os
import shutil
import time
from glob import glob

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    PReLU,
    UpSampling2D,
)
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from util.data_loader import DataLoader, PoolData
from util.generate import denormalize, normalize
from util.layer import create_vgg_19_features_model
from util.logger import create_logger
from util.metric import cal_niqe_tf, cal_psnr_tf, cal_ssim_tf
from util.toml import parse_toml


class SRGAN:
    """
    SRGAN 模型类
    """

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
        self.rdb_num = rdb_num  # 残差块数量
        self.max_workers = max_workers  # 处理图片的最大线程数
        self.data_enhancement_factor = (
            data_enhancement_factor  # 数据增强因子，表示利用随机裁剪和水平翻转来扩充训练数据集的倍数，默认为 1（不进行扩充）
        )
        self.log_interval = log_interval  # 打印日志间隔
        self.save_images_interval = save_images_interval  # 保存图片迭代间隔
        self.save_models_interval = save_models_interval  # 保存模型迭代间隔
        self.save_history_interval = save_history_interval  # 保存历史数据迭代间隔
        self.pretrain_model_path = pretrain_model_path  # 预训练模型路径
        self.use_mixed_float = use_mixed_float  # 是否使用混合精度
        self.use_sn = use_sn  # 是否使用谱归一化
        self.use_ema = use_ema  # 是否使用 EMA

        # 创建日志记录器
        log_dir_path = os.path.join(self.result_path, self.model_name, "logs")
        log_file_name = "%s_train.log" % self.model_name
        self.logger = create_logger(log_dir_path, log_file_name, self.model_name)

        # 创建优化器
        self.pre_optimizer = Adam(1e-4)
        self.gen_optimizer = Adam(1e-4)
        self.dis_optimizer = Adam(1e-4)

        # 检查是否使用混合精度
        self.check_mixed()

        # 损失函数
        self.mse_loss = MeanSquaredError()
        self.mae_loss = MeanAbsoluteError()
        self.bce_loss = BinaryCrossentropy()

        # 损失权重
        self.loss_weights = {"content": 1, "gen": 1e-3}

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

        self.pool_data = PoolData(
            pool_size=10 * self.batch_size, batch_size=self.batch_size
        )

        # 创建 vgg 模型
        self.vgg = self.build_vgg()

        # 创建生成器
        self.generator = self.build_generator()
        # 输出生成器网络结构
        # self.generator.summary()

        # 创建判别器
        self.discriminator = self.build_discriminator()
        # 输出判别器网络结构
        # self.discriminator.summary()

        # # 构建联合模型
        # self.combined = self.build_combined()
        # # 输出联合模型网络结构
        # self.combined.summary()

    def check_mixed(self):
        """
        检查是否使用混合精度
        """
        if self.use_mixed_float:
            mixed_precision.set_global_policy("mixed_float16")
            self.pre_optimizer = mixed_precision.LossScaleOptimizer(self.pre_optimizer)
            self.gen_optimizer = mixed_precision.LossScaleOptimizer(self.gen_optimizer)
            self.dis_optimizer = mixed_precision.LossScaleOptimizer(self.dis_optimizer)

    # def build_combined(self):
    #     """
    #     构建联合模型，将生成器和判别器结合，当训练生成器时，不训练判别器
    #     """
    #     lr_img = Input(shape=self.lr_shape)
    #
    #     # 利用生成器生成假图片
    #     fake_img = self.generator(lr_img)
    #     # 计算假图片的 vgg 特征
    #     fake_features = self.vgg(fake_img)
    #
    #     # 判别器不进行训练
    #     self.discriminator.trainable = False
    #     # 利用判别器计算假图片的真实性
    #     validity = self.discriminator(fake_img)
    #
    #     # 构建联合模型
    #     combined = Model(lr_img, [validity, fake_features])
    #     combined.compile(
    #         optimizer=self.optimizer,
    #         loss=["binary_crossentropy", "mse"],  # 二分类交叉熵 + 均方误差
    #         loss_weights=[1e-3, 1],  # 生成器的损失权重为 0.001，vgg 损失权重为 1
    #     )
    #
    #     return combined

    def build_generator(self):
        """
        构建生成器
        """
        lr_img = Input(shape=[None, None, 3])

        # 第一部分：Conv2D + Relu
        layer_1 = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(lr_img)
        layer_1 = PReLU(shared_axes=[1, 2])(layer_1)

        # 第二部分：rdb_num 个残差块（rdb_num 默认为 16）+ Conv2D + BN，
        # 每个残差块由 Conv2D + BN + ReLU + Conv2D + BN 组成
        layer_2 = layer_1
        for _ in range(self.rdb_num):
            layer_2 = self.residual_block(layer_2, 64)
        layer_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(layer_2)
        layer_2 = BatchNormalization(momentum=0.8)(layer_2)
        layer_2 = Add()([layer_2, layer_1])

        # 第三部分：上采样（单次上采样，图像的分辨率变为原来的 2 倍）
        layer_3 = layer_2
        for _ in range(self.scale_factor // 2):
            layer_3 = self.upsample_block(layer_3)
        sr_img = Conv2D(
            filters=3,
            kernel_size=9,
            strides=1,
            padding="same",
        )(layer_3)
        sr_img = Activation(
            "tanh",
            dtype="float32",
        )(sr_img)

        generator = Model(inputs=lr_img, outputs=sr_img)

        return generator

    def build_discriminator(self):
        """
        构建判别器
        """
        img = Input(shape=self.hr_shape)

        # 判别器结构
        x = self.disc_basic_block(img, filters=64, strides=1, bn=False)
        x = self.disc_basic_block(input=x, filters=64, strides=2, bn=True)

        x = self.disc_basic_block(input=x, filters=128, strides=1, bn=True)
        x = self.disc_basic_block(input=x, filters=128, strides=2, bn=True)

        x = self.disc_basic_block(input=x, filters=256, strides=1, bn=True)
        x = self.disc_basic_block(input=x, filters=256, strides=2, bn=True)

        x = self.disc_basic_block(input=x, filters=512, strides=1, bn=True)
        x = self.disc_basic_block(input=x, filters=512, strides=2, bn=True)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)(x)
        validity = Activation("sigmoid", dtype="float32")(x)

        discriminator = Model(inputs=img, outputs=validity)
        # discriminator.compile(
        #     optimizer=self.dis_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        # )

        return discriminator

    def build_vgg(self):
        """
        构建 vgg 模型
        """
        # vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.hr_shape)
        # vgg.trainable = False
        # for l in vgg.layers:
        #     l.trainable = False
        # vgg.get_layer("block5_conv4").dtype = "float32"

        # return Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)
        return create_vgg_19_features_model(loss_type="srgan")

    # def pretrain_scheduler(self, optimizers, epoch):
    #     """
    #     动态修改学习率
    #     """
    #     if epoch == 100000:
    #         for optimizer in optimizers:
    #             lr = K.get_value(optimizer.lr)
    #             K.set_value(optimizer.lr, lr * 0.1)
    #             self.logger.info("pretrain lr changed to {}".format(lr * 0.1))

    def scheduler(self, optimizers, epoch):
        """
        动态修改学习率
        """
        if epoch == 100000:
            for optimizer in optimizers:
                lr = K.get_value(optimizer.lr)
                K.set_value(optimizer.lr, lr * 0.1)
                self.logger.info("train lr changed to {}".format(lr * 0.1))

    @tf.function
    def pretrain_step(self, lr_img, hr_img):
        """
        单步预训练
        """
        with tf.GradientTape() as tape:
            hr_generated = self.generator(lr_img, training=True)
            loss = self.mse_loss(hr_img, hr_generated)

            if self.use_mixed_float:
                scaled_loss = self.pre_optimizer.get_scaled_loss(loss)
                # # 将数值类型从 float16 转换为 float32
                # loss = tf.cast(loss, dtype=tf.float32)
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
            gen_img = self.generator(lr_img, training=True)

            # ***
            # 判别器
            # ***
            hr_validity = self.discriminator(hr_img, training=True)
            gen_validity = self.discriminator(gen_img, training=True)

            real_labels = tf.ones_like(hr_validity)
            fake_labels = tf.zeros_like(hr_validity)
            # 计算判别器损失
            discriminator_loss = 0.5 * (
                self.bce_loss(real_labels, hr_validity)
                + self.bce_loss(fake_labels, gen_validity)
            )

            # ***
            # 生成器
            # ***
            content_loss = self.content_loss(hr_img, gen_img)  # 计算内容损失
            generator_loss = self.bce_loss(real_labels, gen_validity)  # 计算对抗损失
            generator_total_loss = (
                self.loss_weights["content"] * content_loss
                + self.loss_weights["gen"] * generator_loss
            )  # 计算生成器总损失

            # 若使用混合精度
            if self.use_mixed_float:
                # 将损失值乘以损失标度值
                scaled_disc_loss = self.dis_optimizer.get_scaled_loss(
                    discriminator_loss
                )
                scaled_gen_loss = self.gen_optimizer.get_scaled_loss(
                    generator_total_loss
                )

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

    # def train_old(self):
    #     """
    #     训练模型
    #     """
    #     # 保存模型文件夹路径
    #     save_models_dir_path = os.path.join(self.result_path, self.model_name, "models")
    #     # 若保存模型文件夹不存在，则创建
    #     if not os.path.isdir(save_models_dir_path):
    #         os.makedirs(save_models_dir_path)
    #
    #     # 保存图片文件夹路径
    #     save_images_dir_path = os.path.join(
    #         self.result_path, self.model_name, "images", "test"
    #     )
    #     # 若保存图片文件夹不存在，则创建
    #     if not os.path.isdir(save_images_dir_path):
    #         os.makedirs(save_images_dir_path)
    #
    #     # 保存历史数据文件夹路径
    #     save_history_dir_path = os.path.join(
    #         self.result_path, self.model_name, "images", "history"
    #     )
    #     # 若保存历史数据文件夹不存在，则创建
    #     if not os.path.isdir(save_history_dir_path):
    #         os.makedirs(save_history_dir_path)
    #
    #     # 若初始 epoch 大于 1，则加载模型
    #     if self.init_epoch > 1:
    #         self.generator = tf.keras.models.load_model(
    #             os.path.join(
    #                 save_models_dir_path, "gen_model_epoch_%d" % self.init_epoch
    #             )
    #         )
    #         self.discriminator = tf.keras.models.load_model(
    #             os.path.join(
    #                 save_models_dir_path, "dis_model_epoch_%d" % self.init_epoch
    #             )
    #         )
    #
    #     epoch_list = tf.constant([])
    #     per_loss_list = tf.constant([], dtype=tf.float32)
    #     d_loss_list = tf.constant([], dtype=tf.float32)
    #     d_acc_list = tf.constant([], dtype=tf.float32)
    #     batch_idx_count = tf.constant(
    #         len(self.data_loader.train_data), dtype=tf.float32
    #     )
    #     # 迭代训练
    #     for epoch in range(self.init_epoch, self.epochs + 1):
    #         per_loss_batch_total = tf.constant(0, dtype=tf.float32)
    #         d_loss_batch_total = tf.constant(0, dtype=tf.float32)
    #         d_acc_batch_total = tf.constant(0, dtype=tf.float32)
    #
    #         # 更改学习率
    #         self.scheduler([self.combined, self.discriminator], epoch)
    #
    #         # 加载训练数据集
    #         for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
    #             # 构建标签数组
    #             real_labels = tf.ones([lr_imgs.shape[0]])
    #             fake_labels = tf.zeros([lr_imgs.shape[0]])
    #
    #             fake_imgs = self.generator.predict(lr_imgs)
    #
    #             # -------------------- #
    #             # 训练判别器
    #             # -------------------- #
    #             # self.discriminator.trainable = True
    #             # d_real_loss, d_real_acc = self.discriminator.train_on_batch(
    #             #     hr_imgs, real_labels
    #             # )
    #             # d_fake_loss, d_fake_acc = self.discriminator.train_on_batch(
    #             #     fake_imgs, fake_labels
    #             # )
    #             # d_loss = 0.5 * tf.add(d_real_loss, d_fake_loss)
    #             # d_acc = 0.5 * tf.add(d_real_acc, d_fake_acc)
    #
    #             # 合并真假图片数据
    #             x = tf.concat([hr_imgs, fake_imgs], axis=0)
    #             # 合并真假标签数组
    #             y = tf.concat([real_labels, fake_labels], axis=0)
    #
    #             # 训练判别器
    #             d_loss, d_acc = self.discriminator.train_on_batch(x, y)
    #
    #             # -------------------- #
    #             # 训练生成器
    #             # -------------------- #
    #             # 计算原图的 vgg 特征
    #             hr_features = self.vgg.predict(hr_imgs)
    #
    #             # 训练生成器，此时不训练判别器
    #             # self.discriminator.trainable = False
    #             per_loss, g_loss, con_loss = self.combined.train_on_batch(
    #                 lr_imgs, [real_labels, hr_features]
    #             )
    #
    #             # 统计当前 batch 的总损失和总准确率
    #             per_loss_batch_total += per_loss
    #             d_loss_batch_total += d_loss
    #             d_acc_batch_total += d_acc * 100
    #
    #             # 输出日志
    #             if (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0):
    #                 self.logger.info(
    #                     "epochs: [%d/%d], batches: [%d/%d], d_loss:%.4f, d_acc:%.2f%%, per_loss:%.4f, g_loss:%.4f, con_loss:%.4f"
    #                     % (
    #                         epoch,
    #                         self.epochs,
    #                         batch_idx + 1,
    #                         batch_idx_count,
    #                         d_loss,
    #                         d_acc * 100,
    #                         per_loss,
    #                         g_loss,
    #                         con_loss,
    #                     )
    #                 )
    #
    #         epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
    #         # 计算当前 epoch 下的平均损失和平均准确率
    #         per_loss_list = tf.concat(
    #             [per_loss_list, [per_loss_batch_total / batch_idx_count]], axis=0
    #         )
    #         d_loss_list = tf.concat(
    #             [d_loss_list, [d_loss_batch_total / batch_idx_count]], axis=0
    #         )
    #         d_acc_list = tf.concat(
    #             [d_acc_list, [d_acc_batch_total / batch_idx_count]], axis=0
    #         )
    #
    #         # 保存历史数据
    #         if (epoch) % self.save_history_interval == 0:
    #             self.save_history(
    #                 epoch,
    #                 save_history_dir_path,
    #                 epoch_list,
    #                 per_loss_list,
    #                 d_loss_list,
    #                 d_acc_list,
    #             )
    #
    #         # 保存图片
    #         if epoch % self.save_images_interval == 0:
    #             self.save_images(epoch, save_images_dir_path, 5)
    #
    #         # 保存模型
    #         if epoch % self.save_models_interval == 0:
    #             self.save_models(epoch, save_models_dir_path)

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
        # print("len of train_data):", len(self.data_loader.train_data))
        # print("downsample_mode:", self.downsample_mode)
        config = parse_toml("./config/config.toml")
        degration_config = config["second-order-degradation"]
        for epoch in range(self.init_epoch, self.epochs + 1):
            loss_batch_total = tf.constant(0, dtype=tf.float32)
            batch_start_time = time.time()
            # 加载训练数据集，并训练
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
                # 若为二阶退化模型，需要先对图像进行退化处理，再从数据池中取出数据
                if self.downsample_mode == "second-order":
                    # start_time = datetime.datetime.now()
                    lr_imgs, hr_imgs = self.data_loader.feed_second_order_data(
                        hr_imgs,
                        degration_config,
                        self.train_hr_img_height,
                        self.train_hr_img_width,
                        True,
                        False,
                    )
                    # end_time = datetime.datetime.now()
                    # print("second-order time:", end_time - start_time)
                    lr_imgs, hr_imgs = self.pool_data.get_pool_data(lr_imgs, hr_imgs)
                # 单步预训练
                loss = self.pretrain_step(lr_imgs, hr_imgs)
                loss_batch_total += loss

                # 输出日志
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_end_time = time.time()
                    self.logger.info(
                        "mode: pretrain, epochs: [%d/%d], batches: [%d/%d], loss: %.4f, time: %ds"
                        % (
                            epoch,
                            self.epochs,
                            batch_idx + 1,
                            batch_idx_count,
                            loss,
                            batch_end_time - batch_start_time,
                        )
                    )
                    batch_start_time = time.time()

            # if self.use_ema:

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
                    save_models_dir_path, "gen_model_epoch_%d" % self.init_epoch
                )
            )
            self.discriminator = tf.keras.models.load_model(
                os.path.join(
                    save_models_dir_path, "dis_model_epoch_%d" % self.init_epoch
                )
            )

        config = parse_toml("./config/config.toml")
        degration_config = config["second-order-degradation"]
        epoch_list = tf.constant([])
        g_loss_list = tf.constant([], dtype=tf.float32)
        d_loss_list = tf.constant([], dtype=tf.float32)
        psnr_list = tf.constant([], dtype=tf.float32)
        ssim_list = tf.constant([], dtype=tf.float32)
        niqe_list = tf.constant([], dtype=tf.float32)
        for epoch in range(self.init_epoch, self.epochs + 1):
            g_loss_batch_total = tf.constant(0, dtype=tf.float32)
            d_loss_batch_total = tf.constant(0, dtype=tf.float32)
            # psnr_batch_total = tf.constant(0, dtype=tf.float32)
            # ssim_batch_total = tf.constant(0, dtype=tf.float32)
            batch_idx_count = tf.constant(
                len(self.data_loader.train_data), dtype=tf.float32
            )

            # 修改学习率
            self.scheduler([self.gen_optimizer, self.dis_optimizer], epoch)

            batch_start_time = time.time()
            # 加载训练数据集，并训练
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
                # 若为二阶退化模型，需要先对图像进行退化处理，再从数据池中取出数据
                if self.downsample_mode == "second-order":
                    lr_imgs, hr_imgs = self.data_loader.feed_second_order_data(
                        hr_imgs,
                        degration_config,
                        self.train_hr_img_height,
                        self.train_hr_img_width,
                        True,
                        False,
                    )
                    lr_imgs, hr_imgs = self.pool_data.get_pool_data(lr_imgs, hr_imgs)

                g_loss, d_loss = self.train_step(lr_imgs, hr_imgs)

                # # 计算 PSNR 和 SSIM
                # psnr = cal_psnr_tf(hr_img_list, gen_img_list)
                # ssim = cal_ssim_tf(hr_img_list, gen_img_list)

                g_loss_batch_total += g_loss
                d_loss_batch_total += d_loss
                # psnr_batch_total += psnr
                # ssim_batch_total += ssim

                # 输出日志
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_end_time = time.time()
                    self.logger.info(
                        "mode: train, epochs: [%d/%d], batches: [%d/%d], g_loss: %.4f, d_loss: %.4f, time: %ds"
                        % (
                            epoch,
                            self.epochs,
                            batch_idx + 1,
                            batch_idx_count,
                            g_loss,
                            d_loss,
                            batch_end_time - batch_start_time,
                        )
                    )
                    batch_start_time = time.time()

            # 评估模型
            evalute_config = self.get_evaluate_config()
            for _, config_item in evalute_config.items():
                # 仅对标记的数据集进行评估和统计
                if config_item["is_count"]:
                    psnr, ssim, niqe = self.evaluate(
                        epoch,
                        lr_img_dir=config_item["lr_img_dir"],
                        hr_img_dir=config_item["hr_img_dir"],
                        dataset_name=config_item["dataset_name"],
                    )

            epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
            g_loss_list = tf.concat(
                [g_loss_list, [g_loss_batch_total / batch_idx_count]], axis=0
            )
            d_loss_list = tf.concat(
                [d_loss_list, [d_loss_batch_total / batch_idx_count]], axis=0
            )
            psnr_list = tf.concat([psnr_list, [psnr]], axis=0)
            ssim_list = tf.concat([ssim_list, [ssim]], axis=0)
            niqe_list = tf.concat([niqe_list, [niqe]], axis=0)

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
                    niqe_list,
                )

            # 保存图片
            if epoch % self.save_images_interval == 0:
                self.save_images(epoch, save_images_dir_path, 5)

            # 评估并保存模型
            if epoch % self.save_models_interval == 0:
                for _, config_item in evalute_config.items():
                    # 对非标记的数据集进行评估，避免评估时间过长
                    if not config_item["is_count"]:
                        self.evaluate(
                            epoch,
                            lr_img_dir=config_item["lr_img_dir"],
                            hr_img_dir=config_item["hr_img_dir"],
                            dataset_name=config_item["dataset_name"],
                        )
                self.save_models(epoch, save_models_dir_path)

    def residual_block(self, input, filters):
        """构建残差块

        Args:
            input (_type_): 输入张量
            filters (int): 通道数

        Returns:
            tf.Tensor: 输出张量
        """
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(input)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, input])

        return x

    def upsample_block(self, input):
        """上采样模块

        Args:
            input (tf.Tensor): 原始张量

        Returns:
            tf.Tensor: 上采样后的张量
        """
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(input)
        x = UpSampling2D(size=2)(x)
        x = PReLU(shared_axes=[1, 2])(x)

        return x

    def disc_basic_block(self, input, filters, strides=1, bn=True):
        """判别器基本模块: Conv2D + LeakyReLU + BN

        Args:
            input (tf.tenosr): 输入张量
            filters (int): 通道数
            strides (int, optional): 卷积操作中的步长，默认为 1
            bn (bool, optional): 是否进行批归一化，默认为 True

        Returns:
            tf.tenosr: 输出张量
        """
        x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same")(
            input
        )
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        return x

    # @tf.function
    def valid_step(self, lr_img, hr_img):
        """
        单步验证
        """
        hr_generated = self.generator.predict(lr_img)

        # 反归一化到 [0, 255]，由于生成器最后一层采用了 tanh 激活函数，输出数据被缩放到 [-1, 1]
        hr_img = denormalize(hr_img, (-1, 1))
        hr_generated = denormalize(hr_generated, (-1, 1))

        # 计算 PSNR，SSIM 和 NIQE
        psnr = cal_psnr_tf(hr_img, hr_generated)
        ssim = cal_ssim_tf(hr_img, hr_generated)
        niqe = cal_niqe_tf(hr_generated)

        return psnr, ssim, niqe

    def evaluate(self, epoch, lr_img_dir="", hr_img_dir="", dataset_name="Custom"):
        """
        评估模型，默认在 DIV2K 测试集上进行评估，若要在其他测试集合上评估，需指定 lr_img_dir 和 hr_img_dir 路径
        """
        test_data_len = tf.constant(len(self.data_loader.test_data), dtype=tf.float32)
        psnr_total = tf.constant(0, dtype=tf.float32)
        ssim_total = tf.constant(0, dtype=tf.float32)
        niqe_total = tf.constant(0, dtype=tf.float32)
        if lr_img_dir == "" and hr_img_dir == "":
            for (lr_imgs, hr_imgs) in self.data_loader.test_data:
                # 单步验证
                psnr, ssim, niqe = self.valid_step(lr_imgs, hr_imgs)

                # 统计 PSNR 和 SSIM
                psnr_total += psnr
                ssim_total += ssim
                niqe_total += niqe

            # 输出 PSNR，SSIM 和 NIQE
            self.logger.info(
                "mode: evaluate, epochs: [%d/%d], dataset: %s, PSNR: %.2f, SSIM: %.4f, NIQE: %.2f"
                % (
                    epoch,
                    self.epochs,
                    dataset_name,
                    psnr_total / test_data_len,
                    ssim_total / test_data_len,
                    niqe_total / test_data_len,
                )
            )

            return (
                psnr_total / test_data_len,
                ssim_total / test_data_len,
                niqe_total / test_data_len,
            )
        if lr_img_dir != "" and hr_img_dir != "":
            lr_img_path_list = sorted(glob(os.path.join(lr_img_dir, "*[.png]")))
            hr_img_path_list = sorted(glob(os.path.join(hr_img_dir, "*[.png]")))
            assert len(lr_img_path_list) == len(
                hr_img_path_list
            ), "The length of lr_img_path_list and hr_img_path_list must be same!"
            test_data_len = tf.constant(len(lr_img_path_list), dtype=tf.float32)

            for (lr_img_path, hr_img_path) in list(
                zip(lr_img_path_list, hr_img_path_list)
            ):
                # 读取图片（opencv）
                lr_img = cv.imread(lr_img_path)
                hr_img = cv.imread(hr_img_path)

                # 将图片从 BGR 格式转换为 RGB 格式
                lr_img = cv.cvtColor(lr_img, cv.COLOR_BGR2RGB)
                hr_img = cv.cvtColor(hr_img, cv.COLOR_BGR2RGB)

                # np.unit8 --> tf.uint8
                lr_img = tf.convert_to_tensor(lr_img, dtype=tf.uint8)
                hr_img = tf.convert_to_tensor(hr_img, dtype=tf.uint8)

                # 归一化到 [-1, 1]
                lr_img = normalize(lr_img, (-1, 1))
                hr_img = normalize(hr_img, (-1, 1))

                # 升维
                lr_img = tf.expand_dims(lr_img, axis=0)
                hr_img = tf.expand_dims(hr_img, axis=0)

                # 单步验证
                psnr, ssim, niqe = self.valid_step(lr_img, hr_img)

                # 统计 PSNR 和 SSIM
                psnr_total += psnr
                ssim_total += ssim
                niqe_total += niqe

            # 输出 PSNR，SSIM 和 NIQE
            self.logger.info(
                "mode: evaluate, epochs: [%d/%d], dataset: %s, PSNR: %.2f, SSIM: %.4f, NIQE: %.2f"
                % (
                    epoch,
                    self.epochs,
                    dataset_name,
                    psnr_total / test_data_len,
                    ssim_total / test_data_len,
                    niqe_total / test_data_len,
                )
            )

            return (
                psnr_total / test_data_len,
                ssim_total / test_data_len,
                niqe_total / test_data_len,
            )
        else:
            raise ValueError("The lr_img_dir or hr_img_dir can not be empty!")

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
            dpi=300,
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
        niqe_list,
    ):
        """
        保存历史数据
        """
        fig = plt.figure(figsize=(10, 10))

        # 绘制损失曲线
        ax_1 = plt.subplot(2, 2, 1)
        ax_1.set_title("Train Loss")
        (line_1,) = ax_1.plot(
            epoch_list, g_loss_list, color="deepskyblue", marker=".", label="g_loss"
        )
        (line_2,) = ax_1.plot(
            epoch_list, d_loss_list, color="darksalmon", marker=".", label="d_loss"
        )
        ax_1.set_xlabel("epoch")
        ax_1.set_ylabel("Loss")
        ax_1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_1.legend(handles=[line_1, line_2], loc="upper right")

        # 绘制 PSNR 曲线
        ax_2 = plt.subplot(2, 2, 2)
        ax_2.set_title("PSNR")
        ax_2.plot(epoch_list, psnr_list, color="orange", marker=".", label="PSNR")
        ax_2.set_xlabel("epoch")
        ax_2.set_ylabel("PSNR(dB)")
        ax_2.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 绘制 SSIM 曲线
        ax_3 = plt.subplot(2, 2, 3)
        ax_3.set_title("SSIM")
        ax_3.plot(epoch_list, ssim_list, color="salmon", marker=".", label="SSIM")
        ax_3.set_xlabel("epoch")
        ax_3.set_ylabel("SSIM")
        ax_3.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 绘制 NIQE 曲线
        ax_4 = plt.subplot(2, 2, 4)
        ax_4.set_title("NIQE")
        ax_4.plot(epoch_list, niqe_list, color="purple", marker=".", label="NIQE")
        ax_4.set_xlabel("epoch")
        ax_4.set_ylabel("NIQE")
        ax_4.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_history_dir_path, "train_history_epoch_%d.png" % epoch),
            dpi=300,
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
            lr_img = tf.expand_dims(lr_img, axis=0)
            sr_img = self.generator(lr_img, training=False)
            sr_img = tf.squeeze(sr_img, axis=0)
            lr_img = tf.squeeze(lr_img, axis=0)

            # 反归一化
            lr_img, hr_img, sr_img = (
                denormalize(lr_img, (-1, 1)),
                denormalize(hr_img, (-1, 1)),
                denormalize(sr_img, (-1, 1)),
            )

            axs[i, 0].imshow(lr_img)
            axs[i, 0].axis("off")
            if i == 0:
                axs[i, 0].set_title("Bicubic")

            axs[i, 1].imshow(sr_img)
            axs[i, 1].axis("off")
            if i == 0:
                axs[i, 1].set_title(self.model_name.upper())

            axs[i, 2].imshow(hr_img)
            axs[i, 2].axis("off")
            if i == 0:
                axs[i, 2].set_title("Ground Truth")
        # 保存图片
        fig.savefig(
            os.path.join(save_images_dir_path, "test_epoch_%d.png" % epoch),
            dpi=300,
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

    def content_loss(self, hr_img, hr_generated):
        """
        内容损失
        """
        # 反归一化
        hr_generated = denormalize(hr_generated, normalized_interval=(-1, 1))
        hr_img = denormalize(hr_img, normalized_interval=(-1, 1))

        hr_generated_features = self.vgg(hr_generated) / 12.75
        hr_features = self.vgg(hr_img) / 12.75

        return self.mse_loss(hr_features, hr_generated_features)

    def get_evaluate_config(self):
        """
        获取评估测试集相关配置信息
        """
        config = parse_toml("./config/config.toml")
        evaluate_config = config["evaluate_dataset"]
        return evaluate_config
