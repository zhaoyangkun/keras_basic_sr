import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Input,
    Lambda,
    LeakyReLU,
    PReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


class ESRGAN(object):
    def __init__(
        self,
        model_name,
        result_path,
        train_resource_path,
        test_resource_path,
        epochs,
        init_epoch=0,
        batch_size=4,
        scale_factor=4,
        hr_img_height=128,
        hr_img_width=128,
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
            hr_img_height // scale_factor,
            hr_img_width // scale_factor,
            3,
        )  # 缩放后的图片尺寸
        self.hr_shape = (hr_img_height, hr_img_width, 3)  # 原图尺寸
        self.rdb_num = rdb_num  # 残差块数量
        self.max_workers = max_workers  # 处理图片的最大线程数
        self.data_enhancement_factor = (
            data_enhancement_factor  # 数据增强因子，表示利用随机裁剪和水平翻转来扩充训练数据集的倍数，默认为 1（不进行扩充）
        )
        self.log_interval = log_interval  # 打印日志间隔
        self.save_images_interval = save_images_interval  # 保存图片迭代间隔
        self.save_models_interval = save_models_interval  # 保存模型迭代间隔
        self.save_history_interval = save_history_interval  # 保存历史数据迭代间隔

        self.build_generator()

        # # 创建日志记录器
        # log_dir_path = os.path.join(self.result_path, self.model_name, "logs")
        # log_file_name = "%s_train.log" % self.model_name
        # self.logger = create_logger(log_dir_path, log_file_name, self.model_name)

        # # 创建优化器
        # self.optimizer = Adam(0.0002, 0.5)

        # # 创建 vgg 模型
        # self.vgg = self.build_vgg()

        # # 创建数据集
        # self.data_loader = DataLoader(
        #     self.train_resource_path,
        #     self.test_resource_path,
        #     self.batch_size,
        #     self.hr_shape[0],
        #     self.hr_shape[1],
        #     self.scale_factor,
        #     self.max_workers,
        #     self.data_enhancement_factor,
        # )

        # # 创建生成器
        # self.generator = self.build_generator()
        # # 输出生成器网络结构
        # self.generator.summary()

        # # 创建判别器
        # self.discriminator = self.build_discriminator()
        # # 输出判别器网络结构
        # self.discriminator.summary()

        # # 构建联合模型
        # self.combined = self.build_combined()
        # # 输出联合模型网络结构
        # self.combined.summary()

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
        img = Input(shape=self.hr_shape)

        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[20].output]

        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model

    def preprocess_vgg(self, x):
        """处理 vgg 输入

        Args:
            x (tf.Tensor): 输入
        """
        if isinstance(x, np.ndarray):
            return preprocess_input((x + 1) * 127.5)
        else:
            return Lambda(lambda x: preprocess_input((x + 1) * 127.5))(x)

    def build_generator(self):
        """
        构建生成器
        """

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding="same")(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding="same")(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding="same")(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding="same")(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])

            x5 = Conv2D(64, kernel_size=3, strides=1, padding="same")(x4)
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
        lr_input = Input(shape=self.lr_shape)

        # RRDB 之前
        x_start = Conv2D(64, kernel_size=3, strides=1, padding="same")(lr_input)
        x_start = LeakyReLU(0.5)(x_start)

        # RRDB
        x = RRDB(x_start)

        # RRDB 之后
        x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # 上采样
        for i in range(self.scale_factor // 2):
            x = upsample(x, i + 1)  # 每次上采样，图像尺寸变为原来的两倍

        x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(
            3, kernel_size=3, strides=1, padding="same", activation="tanh"
        )(x)

        model = Model(inputs=lr_input, outputs=hr_output)
        # model.summary()

        return model


# if __name__ == "__main__":
#     esrgan = ESRGAN()
