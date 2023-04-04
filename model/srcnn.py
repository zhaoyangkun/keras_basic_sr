import os
import shutil
import time
from glob import glob

import cv2 as cv
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from util.data_loader import DataLoader, PoolData
from util.data_util import resize
from util.data_util import denormalize, normalize
from util.logger import create_logger
from util.metric import cal_niqe_tf, cal_psnr_tf, cal_ssim_tf
from util.toml import parse_toml


class SRCNN:
    """
    SRCNN 模型类
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
        # self.optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
        self.optimizer = Adam(learning_rate=1e-3)

        # 检查是否使用混合精度
        self.check_mixed()

        # 损失函数
        self.mse_loss = MeanSquaredError()
        # self.mae_loss = MeanAbsoluteError()

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

        # 创建模型
        self.generator = self.build_generator()
        self.generator.summary()

    def check_mixed(self):
        """
        检查是否使用混合精度
        """
        if self.use_mixed_float:
            mixed_precision.set_global_policy("mixed_float16")
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    def build_generator(self):
        """
        构建生成器
        """
        inputs = Input(shape=[None, None, 3])

        x = Conv2D(filters=64, kernel_size=9, padding="same", activation="relu")(inputs)
        x = Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(x)

        outputs = Conv2D(filters=3, kernel_size=5, padding="same", dtype="float32")(x)

        return Model(inputs=inputs, outputs=outputs)

    @tf.function
    def train_step(self, lr_img, hr_img):
        """
        单步训练
        """
        with tf.GradientTape() as tape:
            gen_img = self.generator(lr_img, training=True)
            loss = self.mse_loss(hr_img, gen_img)
            # loss = self.mae_loss(hr_img, gen_img)

            # 若使用混合精度
            if self.use_mixed_float:
                # 将损失值乘以损失标度值
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        # 若使用混合精度，将梯度除以损失标度
        if self.use_mixed_float:
            scaled_gradients = tape.gradient(
                scaled_loss, self.generator.trainable_variables
            )
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        # 不使用混合精度，直接获取梯度
        else:
            gradients = tape.gradient(loss, self.generator.trainable_variables)
        # 更新优化器参数
        self.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return loss

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

        # 若初始 epoch 大于 1，则加载模型
        if self.init_epoch > 1:
            self.generator = tf.keras.models.load_model(
                os.path.join(
                    save_models_dir_path, "gen_model_epoch_%d" % self.init_epoch
                )
            )

        config = parse_toml("./config/config.toml")
        degration_config = config["second-order-degradation"]
        epoch_list = tf.constant([])
        loss_list = tf.constant([], dtype=tf.float32)
        psnr_list = tf.constant([], dtype=tf.float32)
        ssim_list = tf.constant([], dtype=tf.float32)
        niqe_list = tf.constant([], dtype=tf.float32)
        for epoch in range(self.init_epoch, self.epochs + 1):
            loss_batch_total = tf.constant(0, dtype=tf.float32)
            batch_idx_count = tf.constant(
                len(self.data_loader.train_data), dtype=tf.float32
            )
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

                # 由于 SRCNN 模型中没有上采样层，因此需要利用 bicubic 算法对低分辨率图像进行上采样，
                # 以保证低分辨率图像尺寸和原图尺寸一致
                lr_imgs = resize(
                    lr_imgs,
                    self.train_hr_img_width,
                    self.train_hr_img_height,
                    3,
                    "bicubic",
                    normalized_interval=(-1, 1),
                )

                # 单步训练
                loss = self.train_step(lr_imgs, hr_imgs)

                # 统计所有 minibatch 上的总损失
                loss_batch_total += loss

                # 输出日志
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_end_time = time.time()
                    self.logger.info(
                        "mode: train, epochs: [%d/%d], batches: [%d/%d], loss: %.4f, time: %ds"
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
                # else:
                #     self.evaluate(
                #         epoch,
                #         lr_img_dir=config_item["lr_img_dir"],
                #         hr_img_dir=config_item["hr_img_dir"],
                #         dataset_name=config_item["dataset_name"],
                #     )

            # 统计每个 epoch 对应的 loss、psnr、ssim
            epoch_list = tf.concat([epoch_list, [epoch]], axis=0)
            loss_list = tf.concat(
                [loss_list, [loss_batch_total / batch_idx_count]], axis=0
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
                    loss_list,
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

    def valid_step(self, lr_img, hr_img):
        """
        单步验证
        """
        hr_generated = self.generator(lr_img, training=False)

        # 反归一化
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
                # 上采样
                lr_imgs = resize(
                    lr_imgs,
                    self.train_hr_img_width,
                    self.train_hr_img_height,
                    3,
                    "bicubic",
                    normalized_interval=(-1, 1),
                )

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

                # 上采样
                lr_img = resize(
                    lr_img,
                    hr_img.shape[1],
                    hr_img.shape[0],
                    3,
                    "bicubic",
                    normalized_interval=(-1, 1),
                )

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

    def save_history(
        self,
        epoch,
        save_history_dir_path,
        epoch_list,
        loss_list,
        psnr_list,
        ssim_list,
        niqe_list,
    ):
        """
        保存历史数据
        """
        fig = plt.figure(figsize=(10, 10))

        # 绘制 Loss 曲线
        ax_1 = plt.subplot(2, 2, 1)
        ax_1.set_title("Train Loss")
        (_,) = ax_1.plot(
            epoch_list, loss_list, color="deepskyblue", marker=".", label="loss"
        )
        # (line_1,) = ax_1.plot(
        #     epoch_list, loss_list, color="deepskyblue", marker=".", label="loss"
        # )
        ax_1.set_xlabel("epoch")
        ax_1.set_ylabel("Loss")
        ax_1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax_1.set_xlim(0, epoch + 1)
        # ax_1.legend(handles=[line_1], loc="upper right")

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
        fig, axs = plt.subplots(take_num, 4)
        for i, (lr_img, hr_img) in enumerate(test_dataset):
            # 上采样
            lr_img_large = resize(
                lr_img,
                self.train_hr_img_width,
                self.train_hr_img_height,
                3,
                "bicubic",
                normalized_interval=(-1, 1),
            )

            # 利用生成器生成图片
            lr_img_large = tf.expand_dims(lr_img_large, axis=0)
            sr_img = self.generator(lr_img_large, training=False)
            sr_img = tf.squeeze(sr_img, axis=0)
            lr_img_large = tf.squeeze(lr_img_large, axis=0)

            # 反归一化
            lr_img, lr_img_large, hr_img, sr_img = (
                denormalize(lr_img, (-1, 1)),
                denormalize(lr_img_large, (-1, 1)),
                denormalize(hr_img, (-1, 1)),
                denormalize(sr_img, (-1, 1)),
            )

            axs[i, 0].imshow(lr_img)
            axs[i, 0].axis("off")
            if i == 0:
                axs[i, 0].set_title("Bicubic")

            axs[i, 1].imshow(lr_img_large)
            axs[i, 1].axis("off")
            if i == 0:
                axs[i, 1].set_title("Bicubic Large")

            axs[i, 2].imshow(sr_img)
            axs[i, 2].axis("off")
            if i == 0:
                axs[i, 2].set_title(self.model_name.upper())

            axs[i, 3].imshow(hr_img)
            axs[i, 3].axis("off")
            if i == 0:
                axs[i, 3].set_title("Ground Truth")
        # 保存图片
        fig.savefig(
            os.path.join(save_images_dir_path, "test_epoch_%d.png" % epoch),
            dpi=300,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close(fig)

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

    def get_evaluate_config(self):
        """
        获取评估测试集相关配置信息
        """
        config = parse_toml("./config/config.toml")
        evaluate_config = config["evaluate_dataset"]
        return evaluate_config
