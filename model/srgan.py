import os
import shutil

import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    PReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from util.data_loader import DataLoader
from util.logger import create_logger


class SRGAN(object):
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
    ):
        self.model_name = model_name  # 模型名称
        self.result_path = result_path  # 结果保存路径
        self.train_resource_path = train_resource_path  # 训练图片资源路径
        self.test_resource_path = test_resource_path  # 测试图片资源路径
        self.epochs = epochs  # 训练轮数
        self.init_epoch = init_epoch  # 初始化训练轮数
        self.batch_size = batch_size  # 批次大小
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

        # 创建日志记录器
        log_dir_path = os.path.join(self.result_path, self.model_name, "logs")
        log_file_name = "%s_train.log" % self.model_name
        self.logger = create_logger(log_dir_path, log_file_name, self.model_name)

        # 创建优化器
        self.optimizer = Adam(0.0002, 0.5)

        # 创建 vgg 模型
        self.vgg = self.build_vgg()

        # 创建数据集
        self.data_loader = DataLoader(
            self.train_resource_path,
            self.test_resource_path,
            self.batch_size,
            self.train_hr_img_height,
            self.train_hr_img_width,
            self.valid_hr_img_height,
            self.valid_hr_img_width,
            self.scale_factor,
            self.max_workers,
            self.data_enhancement_factor,
        )

        # 创建生成器
        self.generator = self.build_generator()
        # 输出生成器网络结构
        self.generator.summary()

        # 创建判别器
        self.discriminator = self.build_discriminator()
        # 输出判别器网络结构
        self.discriminator.summary()

        # 构建联合模型
        self.combined = self.build_combined()
        # 输出联合模型网络结构
        self.combined.summary()

    def build_combined(self):
        """
        构建联合模型，将生成器和判别器结合，当训练生成器时，不训练判别器
        """
        lr_img = Input(shape=self.lr_shape)

        # 利用生成器生成假图片
        fake_img = self.generator(lr_img)
        # 计算假图片的 vgg 特征
        fake_features = self.vgg(fake_img)

        # 判别器不进行训练
        self.discriminator.trainable = False
        # 利用判别器计算假图片的真实性
        validity = self.discriminator(fake_img)

        # 构建联合模型
        combined = Model(lr_img, [validity, fake_features])
        combined.compile(
            optimizer=self.optimizer,
            loss=["binary_crossentropy", "mse"],  # 二分类交叉熵 + 均方误差
            loss_weights=[1e-3, 1],  # 生成器的损失权重为 0.001，vgg 损失权重为 1
        )

        return combined

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

        # 第三部分：上采样两次，分辨率变为原来的 4 倍（单次上采样，图像的分辨率变为原来的 2 倍）
        layer_3 = self.upsample_block(layer_2)
        layer_3 = self.upsample_block(layer_3)
        sr_img = Conv2D(
            filters=3, kernel_size=9, strides=1, padding="same", activation="tanh"
        )(layer_3)

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
        validity = Dense(1, activation="sigmoid")(x)

        discriminator = Model(inputs=img, outputs=validity)
        discriminator.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        return discriminator

    def build_vgg(self):
        """
        构建 vgg 模型
        """
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.hr_shape)
        vgg.trainable = False
        for l in vgg.layers:
            l.trainable = False

        return Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

    def scheduler(self, models, epoch):
        """动态修改学习率

        Args:
            models (list): 模型列表
            epoch (int): 当前 epoch
        """
        if epoch % 500 == 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))

    def train(self):
        """
        训练模型
        """
        # 保存模型文件夹路径
        save_models_dir_path = os.path.join(self.result_path, self.model_name, "models")
        # 若保存模型文件夹不存在，则创建
        if not os.path.isdir(save_models_dir_path):
            os.makedirs(save_models_dir_path)

        # 保存图片文件夹路径
        save_images_dir_path = os.path.join(
            self.result_path, self.model_name, "images", "test"
        )
        # 若保存图片文件夹不存在，则创建
        if not os.path.isdir(save_images_dir_path):
            os.makedirs(save_images_dir_path)

        # 保存历史数据文件夹路径
        save_history_dir_path = os.path.join(
            self.result_path, self.model_name, "images", "history"
        )
        # 若保存历史数据文件夹不存在，则创建
        if not os.path.isdir(save_history_dir_path):
            os.makedirs(save_history_dir_path)

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
        per_loss_list = tf.constant([], dtype=tf.float32)
        d_loss_list = tf.constant([], dtype=tf.float32)
        d_acc_list = tf.constant([], dtype=tf.float32)
        batch_idx_count = tf.constant(
            len(self.data_loader.train_data), dtype=tf.float32
        )
        # 迭代训练
        for epoch in range(self.init_epoch, self.epochs + 1):
            per_loss_batch_total = tf.constant(0, dtype=tf.float32)
            d_loss_batch_total = tf.constant(0, dtype=tf.float32)
            d_acc_batch_total = tf.constant(0, dtype=tf.float32)

            # 更改学习率
            self.scheduler([self.combined, self.discriminator], epoch)

            # 加载训练数据集
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.data_loader.train_data):
                # 构建标签数组
                real_labels = tf.ones([lr_imgs.shape[0]])
                fake_labels = tf.zeros([lr_imgs.shape[0]])

                fake_imgs = self.generator.predict(lr_imgs)

                # -------------------- #
                # 训练判别器
                # -------------------- #
                # self.discriminator.trainable = True
                # d_real_loss, d_real_acc = self.discriminator.train_on_batch(
                #     hr_imgs, real_labels
                # )
                # d_fake_loss, d_fake_acc = self.discriminator.train_on_batch(
                #     fake_imgs, fake_labels
                # )
                # d_loss = 0.5 * tf.add(d_real_loss, d_fake_loss)
                # d_acc = 0.5 * tf.add(d_real_acc, d_fake_acc)

                # 合并真假图片数据
                x = tf.concat([hr_imgs, fake_imgs], axis=0)
                # 合并真假标签数组
                y = tf.concat([real_labels, fake_labels], axis=0)

                # 训练判别器
                d_loss, d_acc = self.discriminator.train_on_batch(x, y)

                # -------------------- #
                # 训练生成器
                # -------------------- #
                # 计算原图的 vgg 特征
                hr_features = self.vgg.predict(hr_imgs)

                # 训练生成器，此时不训练判别器
                # self.discriminator.trainable = False
                per_loss, g_loss, con_loss = self.combined.train_on_batch(
                    lr_imgs, [real_labels, hr_features]
                )

                # 统计当前 batch 的总损失和总准确率
                per_loss_batch_total += per_loss
                d_loss_batch_total += d_loss
                d_acc_batch_total += d_acc * 100

                # 输出日志
                if (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0):
                    self.logger.info(
                        "epochs: [%d/%d], batches: [%d/%d], d_loss:%.4f, d_acc:%.2f%%, per_loss:%.4f, g_loss:%.4f, con_loss:%.4f"
                        % (
                            epoch,
                            self.epochs,
                            batch_idx + 1,
                            batch_idx_count,
                            d_loss,
                            d_acc * 100,
                            per_loss,
                            g_loss,
                            con_loss,
                        )
                    )

            epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
            # 计算当前 epoch 下的平均损失和平均准确率
            per_loss_list = tf.concat(
                [per_loss_list, [per_loss_batch_total / batch_idx_count]], axis=0
            )
            d_loss_list = tf.concat(
                [d_loss_list, [d_loss_batch_total / batch_idx_count]], axis=0
            )
            d_acc_list = tf.concat(
                [d_acc_list, [d_acc_batch_total / batch_idx_count]], axis=0
            )

            # 保存历史数据
            if (epoch) % self.save_history_interval == 0:
                self.save_history(
                    epoch,
                    save_history_dir_path,
                    epoch_list,
                    per_loss_list,
                    d_loss_list,
                    d_acc_list,
                )

            # 保存图片
            if epoch % self.save_images_interval == 0:
                self.save_images(epoch, save_images_dir_path, 5)

            # 保存模型
            if epoch % self.save_models_interval == 0:
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

    def save_images(self, epoch, save_images_dir_path, take_num=5):
        """保存图片

        Args:
            epoch (int): 当前训练的轮数
            save_images_dir_path (str): 保存图片的根目录
            take_num (int, optional): 输出图片的行数. Defaults to 5.
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
                axs[i, 1].set_title("SRGAN")

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

    def save_models(self, epoch, save_models_dir_path):
        """保存模型

        Args:
            epoch (int): 当前训练轮数
            save_weights_dir_path (str): 保存模型文件夹路径
        """
        # 删除原先的模型文件，仅保存最新的模型
        file_name_list = os.listdir(save_models_dir_path)
        for file_name in file_name_list:
            file_path = os.path.join(save_models_dir_path, file_name)
            if os.path.exists(file_path):
                shutil.rmtree(file_path)

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

    def save_history(
        self,
        epoch,
        save_history_dir_path,
        epoch_list,
        g_loss_list,
        d_loss_list,
        d_acc_list,
    ):
        """保存历史数据（损失、准确率）

        Args:
            epoch_list (list): 训练轮数列表
            g_loss_list (list): 生成器损失列表
            d_loss_list (list): 判别器损失列表
            d_acc_list (list): 判别器准确率列表
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # 绘制损失曲线
        axes[0].set_title("Loss")
        (line_1,) = axes[0].plot(
            epoch_list, g_loss_list, color="deepskyblue", marker=".", label="g_loss"
        )
        (line_2,) = axes[0].plot(
            epoch_list, d_loss_list, color="darksalmon", marker=".", label="d_loss"
        )
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend(handles=[line_1, line_2], loc="lower right")

        # 绘制准确率曲线
        axes[1].set_title("Accuracy")
        (line_3,) = axes[1].plot(
            epoch_list, d_acc_list, color="orange", marker=".", label="d_acc"
        )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy(%)")
        axes[1].legend(handles=[line_3], loc="lower right")

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_history_dir_path, "history_epoch_%d.png" % epoch),
            dpi=500,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

    # def validate_from_saved_weights(self, weights_path, lr_img):
    #     """
    #     从权重文件中加载模型，并进行验证
    #     """
    #     # 加载模型
    #     self.generator.load_weights(weights_path)
    #     sr_img = self.generator.predict(tf.expand_dims(lr_img, 0))
    #     sr_img = tf.squeeze(sr_img, axis=0)
    #     sr_img = tf.cast((sr_img + 1) * 127.5, dtype=tf.uint8).numpy()
    #     # sr_img = tf.cast((lr_img + 1) * 127.5, dtype=tf.uint8).numpy()

    #     plt.imsave("./image/test_sr.png", sr_img)
