import os
import random
from glob import glob

import tensorflow as tf

from util.data_util import (
    USMSharp,
    filter2D,
    generate_kernel,
    generate_sinc_kernel,
    random_add_gaussian_noise,
    random_add_poisson_noise,
)
from util.toml import parse_toml


class DataLoader:
    """
    数据加载类
    """

    def __init__(
        self,
        train_resource_path,
        test_resource_path,
        batch_size=4,
        downsample_mode="bicubic",
        train_hr_img_height=128,
        train_hr_img_width=128,
        valid_hr_img_height=128,
        valid_hr_img_width=128,
        scale_factor=4,
        max_workers=4,
        data_enhancement_factor=1,
    ):
        self.train_resource_path = train_resource_path  # 训练图片资源路径
        self.test_resource_path = test_resource_path  # 测试图片资源路径
        self.batch_size = batch_size  # 单次训练的图片数量
        self.downsample_mode = downsample_mode  # 下采样模式
        self.train_hr_img_height = train_hr_img_height  # 训练过程中原图高度
        self.train_hr_img_width = train_hr_img_width  # 训练过程中原图宽度
        self.valid_hr_img_height = valid_hr_img_height  # 验证过程中原图高度
        self.valid_hr_img_width = valid_hr_img_width  # 验证过程中原图宽度
        self.scale_factor = scale_factor  # 下采样倍数
        self.max_workers = max_workers  # 多线程最大线程数
        self.data_enhancement_factor = (
            data_enhancement_factor  # 数据增强因子，表示利用随机裁剪和水平翻转来扩充训练数据集的倍数，默认为 1（不进行扩充）
        )
        self.train_data = None  # 训练数据
        self.test_data = None  # 测试数据

        self.create_dataset()  # 构建数据集

    def create_dataset(self):
        """
        构建数据集
        """
        print(
            "\n", "**********" * 2 + " start creating train dataset " + "**********" * 2
        )
        # 获取所有训练图片路径
        ori_train_resource_path_list = sorted(
            glob(os.path.join(self.train_resource_path, "*[.png]"))
        )
        train_resource_path_list = []
        # 数据增强
        for _ in range(self.data_enhancement_factor):
            train_resource_path_list += ori_train_resource_path_list
        assert self.batch_size <= len(
            train_resource_path_list
        ), "batch_size must be <= train_data_num"
        # # 处理图片
        # train_lr_img_list, train_hr_img_list = self.process_img_data(
        #     train_resource_path_list,
        #     self.train_hr_img_height,
        #     self.train_hr_img_width,
        #     mode="train",
        # )
        # # 构建训练数据集
        # self.train_data = (
        #     tf.data.Dataset.from_tensor_slices((train_lr_img_list, train_hr_img_list))
        #     .shuffle(len(train_lr_img_list))  # 打乱数据
        #     .batch(self.batch_size)  # 批次大小
        #     .prefetch(tf.data.experimental.AUTOTUNE)  # 预存数据来提升性能
        # )

        self.train_data = (
            tf.data.Dataset.from_tensor_slices((train_resource_path_list))
            .map(
                lambda image_path: self.process_img_data_worker(
                    image_path, self.train_hr_img_height, self.train_hr_img_width
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .shuffle(len(train_resource_path_list))
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        print(
            "\n", "**********" * 2 + " start creating test dataset " + "**********" * 2
        )
        # 获取所有测试图片路径
        test_resource_path_list = sorted(
            glob(os.path.join(self.test_resource_path, "*[.png]"))
        )
        assert self.batch_size <= len(
            test_resource_path_list
        ), "batch_size must be <= test_data_num"
        # # 处理图片
        # test_lr_img_list, test_hr_img_list = self.process_img_data(
        #     test_resource_path_list,
        #     self.valid_hr_img_height,
        #     self.valid_hr_img_width,
        #     is_random_flip=False,
        #     is_random_crop=False,
        #     is_random_rot=False,
        #     is_center_crop=True,
        #     mode="test",
        # )
        # # 构建测试数据集
        # self.test_data = (
        #     tf.data.Dataset.from_tensor_slices((test_lr_img_list, test_hr_img_list))
        #     .batch(self.batch_size)
        #     .prefetch(tf.data.experimental.AUTOTUNE)
        # )
        self.test_data = (
            tf.data.Dataset.from_tensor_slices((test_resource_path_list))
            .map(
                lambda image_path: self.process_img_data_worker(
                    image_path,
                    self.valid_hr_img_height,
                    self.valid_hr_img_width,
                    False,
                    False,
                    False,
                    True,
                    "test",
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    # def process_img_data(
    #     self,
    #     resource_path_list,
    #     hr_img_height,
    #     hr_img_width,
    #     is_random_flip=True,
    #     is_random_crop=True,
    #     is_random_rot=True,
    #     is_center_crop=False,
    #     mode="train",
    # ):
    #     """处理图片数据

    #     Args:
    #         resource_path_list (_type_): 图片路径列表
    #         is_random_flip (bool, optional): 是否随机翻转. Defaults to True.
    #         is_random_crop (bool, optional): 是否随机裁剪. Defaults to True.
    #         is_center_crop (bool, optional): 是否中心裁剪. Defaults to False.

    #     Raises:
    #         Exception: 图片格式错误

    #     Returns:
    #         tf.Tensor, tf.Tensor: 下采样图片列表，原始图片列表
    #     """
    #     if (mode == "train" and self.downsample_mode == "bicubic") or (mode == "test"):
    #         # 下采样图片列表
    #         lr_img_list = tf.constant(
    #             0,
    #             shape=[
    #                 0,
    #                 hr_img_height // self.scale_factor,
    #                 hr_img_width // self.scale_factor,
    #                 3,
    #             ],
    #             dtype=tf.float32,
    #         )
    #         # 原图列表
    #         hr_img_list = tf.constant(
    #             0,
    #             shape=[0, hr_img_height, hr_img_width, 3],
    #             dtype=tf.float32,
    #         )
    #     elif mode == "train" and self.downsample_mode == "second-order":
    #         # 下采样图片列表
    #         lr_img_list = tf.constant(
    #             0,
    #             shape=[0, 400, 400, 3],
    #             dtype=tf.float32,
    #         )
    #         # 原图列表
    #         hr_img_list = tf.constant(
    #             0,
    #             shape=[0, 400, 400, 3],
    #             dtype=tf.float32,
    #         )
    #     else:
    #         raise ValueError("Unsupported image process mode!")

    #     # 多线程处理图片
    #     pbar = tqdm(total=len(resource_path_list))  # 创建进度条
    #     with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
    #         for (lr_img, hr_img) in pool.map(
    #             self.process_img_data_worker,
    #             resource_path_list,
    #             [hr_img_height] * len(resource_path_list),
    #             [hr_img_width] * len(resource_path_list),
    #             [is_random_flip] * len(resource_path_list),
    #             [is_random_crop] * len(resource_path_list),
    #             [is_random_rot] * len(resource_path_list),
    #             [is_center_crop] * len(resource_path_list),
    #             [mode] * len(resource_path_list),
    #         ):
    #             # 扩大维度
    #             lr_img = tf.expand_dims(lr_img, axis=0)
    #             hr_img = tf.expand_dims(hr_img, axis=0)
    #             # 拼接张量
    #             lr_img_list = tf.concat([lr_img_list, lr_img], axis=0)
    #             hr_img_list = tf.concat([hr_img_list, hr_img], axis=0)
    #             # 更新进度条
    #             pbar.update(1)
    #         pbar.close()  # 关闭进度条

    #     return lr_img_list, hr_img_list

    def process_img_data_worker(
        self,
        img_path,
        hr_img_height,
        hr_img_width,
        is_random_flip=True,
        is_random_crop=True,
        is_random_rot=True,
        is_center_crop=False,
        mode="train",
    ):
        """
        多线程处理图片工作函数
        """
        # 读取图片
        hr_img = tf.io.read_file(img_path)

        hr_img = tf.image.decode_png(hr_img, channels=3)

        # # 根据图片格式进行解码
        # if img_type == "png":
        #     hr_img = tf.image.decode_png(hr_img, channels=3)
        # elif img_type == "jpg" or img_type == "jpeg":
        #     hr_img = tf.image.decode_jpeg(hr_img, channels=3)
        # elif img_type == "bmp":
        #     hr_img = tf.image.decode_bmp(hr_img, channels=3)
        # elif img_type == "gif":
        #     hr_img = tf.image.decode_gif(hr_img)
        # else:
        #     raise Exception("Unknown image type!")

        if (mode == "train" and self.downsample_mode == "bicubic") or (mode == "test"):
            # 随机剪裁
            if is_random_crop:
                hr_img = tf.image.random_crop(hr_img, [hr_img_height, hr_img_width, 3])

            # 中心裁剪
            size = tf.shape(hr_img)
            height, width = size[0], size[1]
            if is_center_crop:
                offset_height = height // 2 - hr_img_height // 2
                offset_width = width // 2 - hr_img_width // 2
                hr_img = tf.image.crop_to_bounding_box(
                    hr_img,
                    offset_height,
                    offset_width,
                    hr_img_height,
                    hr_img_width,
                )

            # 随机水平翻转
            if is_random_flip:
                hr_img = tf.image.random_flip_left_right(hr_img)

            # 随机旋转 90 度（逆时针）
            if is_random_rot and tf.random.uniform([], 0, 1) > 0.5:
                hr_img = tf.image.rot90(hr_img)

            # 归一化到 [-1, 1]
            hr_img = tf.cast(hr_img, tf.float32) / 127.5 - 1
            # hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)

            # 双三次下采样（基于 tensorflow）
            lr_img = tf.image.resize(
                hr_img,
                [
                    hr_img_height // self.scale_factor,
                    hr_img_width // self.scale_factor,
                ],
                method=tf.image.ResizeMethod.BICUBIC,
            )
        elif mode == "train" and self.downsample_mode == "second-order":
            # 归一化到 [0, 1]
            hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)

            # 随机水平翻转
            if is_random_flip:
                hr_img = tf.image.random_flip_left_right(hr_img)

            # 随机旋转 90 度（逆时针）
            if is_random_rot and tf.random.uniform([]) > 0.5:
                hr_img = tf.image.rot90(hr_img)

            # 边缘填充
            size = tf.shape(hr_img)
            height, width = size[0], size[1]
            crop_pad_size = 400
            if height < crop_pad_size or width < crop_pad_size:
                pad_h = tf.maximum(0, crop_pad_size - height)
                pad_w = tf.maximum(0, crop_pad_size - width)

                padding = [[0, pad_h], [0, pad_w], [0, 0]]
                hr_img = tf.pad(hr_img, padding, "REFLECT")

            # 裁剪图片大小为 400 * 400
            size = tf.shape(hr_img)
            height, width = size[0], size[1]
            if height > crop_pad_size or width > crop_pad_size:
                if (height - crop_pad_size) <= 0:
                    top = 0
                else:
                    if is_random_crop:
                        top = tf.random.uniform(
                            [], 0, height - crop_pad_size, dtype=tf.int32
                        )
                    if is_center_crop:
                        top = height // 2 - crop_pad_size // 2

                if (width - crop_pad_size) <= 0:
                    left = 0
                else:
                    if is_random_crop:
                        left = tf.random.uniform(
                            [], 0, width - crop_pad_size, dtype=tf.int32
                        )
                    if is_center_crop:
                        left = width // 2 - crop_pad_size // 2

                hr_img = tf.image.crop_to_bounding_box(
                    hr_img, top, left, crop_pad_size, crop_pad_size
                )
            # # 升维
            # hr_img = tf.expand_dims(hr_img, axis=0)
            # # 读取二阶退化模型相关参数
            # config = parse_toml("./config/config.toml")
            # degration_config = config["second-order-degradation"]
            # # 二阶退化
            # lr_img, hr_img = self.feed_second_order_data(
            #     hr_img,
            #     degration_config,
            #     self.train_hr_img_height,
            #     self.train_hr_img_width,
            #     True,
            #     False,
            # )
            # # 降维
            # lr_img, hr_img = tf.squeeze(lr_img, axis=0), tf.squeeze(hr_img, axis=0)
            lr_img = tf.zeros_like(hr_img)

        else:
            raise ValueError("Unsupported image process mode!")

        return lr_img, hr_img

    def degradation(
        self,
        img,
        batch,
        ori_height,
        ori_width,
        blur_kernel,
        sinc_kernel,
        opts_dict,
        stage="first",
    ):
        """
        图像退化
        """
        # 模糊
        if (opts_dict["blur_prob"] == 1.0) or (
            tf.random.uniform([]) < opts_dict["blur_prob"]
        ):
            img = filter2D(img, blur_kernel)

        # 随机调整图像大小
        updown_type = random.choices(["up", "down", "keep"], opts_dict["resize_prob"])[
            0
        ]
        if updown_type == "up":
            scale = tf.random.uniform([], 1, opts_dict["resize_range"][1])
        elif updown_type == "down":
            scale = tf.random.uniform([], opts_dict["resize_range"][0], 1)
        else:
            scale = tf.constant(1.0, dtype=tf.float32)
        if scale != 1:
            mode = random.choice(["area", "bilinear", "bicubic"])
            if stage == "first":
                resize_height = tf.cast(
                    scale * tf.cast(ori_height, tf.float32), tf.int32
                )
                resize_width = tf.cast(scale * tf.cast(ori_width, tf.float32), tf.int32)
            elif stage == "second":
                resize_height = tf.cast(
                    tf.cast(ori_height, tf.float32) / self.scale_factor * scale,
                    tf.int32,
                )
                resize_width = tf.cast(
                    tf.cast(ori_width, tf.float32) / self.scale_factor * scale, tf.int32
                )
            img = tf.image.resize(img, [resize_height, resize_width], method=mode)

        # 添加噪声
        gray_noise_prob = opts_dict["gray_noise_prob"]
        if tf.random.uniform([]) < opts_dict["gaussian_noise_prob"]:
            img = random_add_gaussian_noise(
                img, sigma_range=opts_dict["noise_range"], gray_prob=gray_noise_prob
            )
        else:
            img = random_add_poisson_noise(
                img,
                scale_range=opts_dict["poisson_scale_range"],
                gray_prob=gray_noise_prob,
            )

        # 第一阶段，只需压缩图像
        if stage == "first":
            # 压缩图像
            img = tf.clip_by_value(img, 0, 1)
            img = [
                tf.image.random_jpeg_quality(
                    img[i], opts_dict["jpeg_range"][0], opts_dict["jpeg_range"][1]
                )
                for i in range(0, batch)
            ]
            img = tf.convert_to_tensor(img)
        # 第二阶段，需要进行恢复图像大小 + sinc 滤波 + 压缩图像
        elif stage == "second":
            if tf.random.uniform([]) < 0.5:
                # 随机下采样
                mode = random.choice(["area", "bilinear", "bicubic"])

                resize_height = ori_height // self.scale_factor
                resize_width = ori_width // self.scale_factor

                img = tf.image.resize(img, [resize_height, resize_width], method=mode)

                # sinc 滤波
                img = filter2D(img, sinc_kernel)

                # 压缩图像
                img = tf.clip_by_value(img, 0, 1)
                img = [
                    tf.image.random_jpeg_quality(
                        img[i],
                        opts_dict["jpeg_range"][0],
                        opts_dict["jpeg_range"][1],
                    )
                    for i in range(0, batch)
                ]
                img = tf.convert_to_tensor(img)
            else:
                # 压缩图像
                img = tf.clip_by_value(img, 0, 1)
                img = [
                    tf.image.random_jpeg_quality(
                        img[i],
                        opts_dict["jpeg_range"][0],
                        opts_dict["jpeg_range"][1],
                    )
                    for i in range(0, batch)
                ]
                img = tf.convert_to_tensor(img)

                # 随机下采样
                mode = random.choice(["area", "bilinear", "bicubic"])
                resize_height = ori_height // self.scale_factor
                resize_width = ori_width // self.scale_factor
                img = tf.image.resize(img, [resize_height, resize_width], method=mode)

                # sinc 滤波
                img = filter2D(img, sinc_kernel)

        else:
            raise ValueError("Unsupported degradation stage!")

        return img

    def random_crop(self, hr_imgs, lr_imgs, hr_img_height, hr_img_width):
        """
        随机裁剪
        """
        _, ori_lr_height, ori_lr_width, _ = lr_imgs.shape

        lr_img_height = hr_img_height // self.scale_factor
        lr_img_width = hr_img_width // self.scale_factor

        lr_top = random.randint(0, ori_lr_height - lr_img_height)
        lr_left = random.randint(0, ori_lr_width - lr_img_width)
        lr_imgs = tf.image.crop_to_bounding_box(
            lr_imgs, lr_top, lr_left, lr_img_height, lr_img_width
        )

        hr_top, hr_left = int(lr_top * self.scale_factor), int(
            lr_left * self.scale_factor
        )
        hr_imgs = tf.image.crop_to_bounding_box(
            hr_imgs, hr_top, hr_left, hr_img_height, hr_img_width
        )

        return hr_imgs, lr_imgs

    def center_crop(self, hr_imgs, lr_imgs, hr_img_height, hr_img_width):
        """
        中心裁剪
        """
        _, ori_lr_height, ori_lr_width, _ = lr_imgs.shape

        lr_img_height = hr_img_height // self.scale_factor
        lr_img_width = hr_img_width // self.scale_factor

        lr_top = ori_lr_height // 2 - lr_img_height // 2
        lr_left = ori_lr_width // 2 - lr_img_width // 2
        lr_imgs = tf.image.crop_to_bounding_box(
            lr_imgs, lr_top, lr_left, lr_img_height, lr_img_width
        )

        hr_top, hr_left = int(lr_top * self.scale_factor), int(
            lr_left * self.scale_factor
        )
        hr_imgs = tf.image.crop_to_bounding_box(
            hr_imgs, hr_top, hr_left, hr_img_height, hr_img_width
        )

        return hr_imgs, lr_imgs

    def feed_second_order_data(
        self,
        hr_img,
        degration_config,
        crop_img_height,
        crop_img_width,
        is_random_crop,
        is_center_crop,
        is_usm=True,
    ):
        """
        获取二阶退化数据
        """
        # 创建相关核
        first_blur_kernels = tf.constant(0, shape=[0, 21, 21], dtype=tf.float32)
        second_blur_kernels = tf.constant(0, shape=[0, 21, 21], dtype=tf.float32)
        sinc_kernels = tf.constant(0, shape=[0, 21, 21], dtype=tf.float32)
        for _ in range(tf.shape(hr_img)[0]):
            first_blur_kernel = generate_kernel(degration_config["kernel_props_1"])
            second_blur_kernel = generate_kernel(degration_config["kernel_props_2"])
            sinc_kernel = generate_sinc_kernel(degration_config["final_sinc_prob"])
            first_blur_kernels = tf.concat(
                [first_blur_kernels, tf.expand_dims(first_blur_kernel, axis=0)], axis=0
            )
            second_blur_kernels = tf.concat(
                [second_blur_kernels, tf.expand_dims(second_blur_kernel, axis=0)],
                axis=0,
            )
            sinc_kernels = tf.concat(
                [sinc_kernels, tf.expand_dims(sinc_kernel, axis=0)], axis=0
            )

        # 锐化
        if is_usm:
            usm_sharpener = USMSharp()
            hr_img = usm_sharpener.sharp(hr_img)

        # 获取最初图像的数量、高和宽
        batch, ori_height, ori_width, _ = tf.shape(hr_img)

        # ----------------------- 第一阶段退化 ----------------------- #
        lr_img = self.degradation(
            hr_img,
            batch,
            ori_height,
            ori_width,
            first_blur_kernels,
            None,
            degration_config["feed_props_1"],
            stage="first",
        )

        # ----------------------- 第二阶段退化 ----------------------- #
        lr_img = self.degradation(
            lr_img,
            batch,
            ori_height,
            ori_width,
            second_blur_kernels,
            sinc_kernels,
            degration_config["feed_props_2"],
            stage="second",
        )

        # 随机裁剪
        if is_random_crop:
            hr_img, lr_img = self.random_crop(
                hr_img, lr_img, crop_img_height, crop_img_width
            )

        # 中心裁剪
        if is_center_crop:
            hr_img, lr_img = self.center_crop(
                hr_img, lr_img, crop_img_height, crop_img_width
            )

        # 将归一化区间从 [0, 1] 调整到 [-1, 1]
        hr_img = tf.clip_by_value(tf.math.round(hr_img * 255), 0, 255) / 127.5 - 1
        lr_img = tf.clip_by_value(tf.math.round(lr_img * 255), 0, 255) / 127.5 - 1

        return lr_img, hr_img

    def get_image_type(self, img_path):
        """
        获取图片类型
        """
        return tf.strings.split(img_path, ".")[-1]


class PoolData:
    """
    数据池
    """

    def __init__(self, pool_size, batch_size):
        self.pool_size = pool_size  # 数据池大小
        self.idx = list(range(self.pool_size))  # 下标数组
        self.queue_gt = None  # 原始图像队列
        self.queue_lr = None  # 压缩图像队列
        self.batch_size = batch_size  # 批次大小

        if not pool_size % batch_size == 0:  # 校验数据池大小能否被批次大小整除
            raise TypeError(
                f"pool_size ({pool_size}) % batch_size ({batch_size}) should be 0"
            )

    def get_pool_data(self, new_lr_imgs, new_hr_imgs):
        # 若数据池为空，则初始化数据池
        if self.queue_gt is None:
            self.queue_lr = new_lr_imgs
            self.queue_gt = new_hr_imgs

            return (
                new_lr_imgs,
                new_hr_imgs,
            )
        # 数据池已满
        elif self.queue_gt.shape[0] == self.pool_size:
            # 打乱数据
            self.idx = tf.random.shuffle(self.idx)
            self.queue_lr = tf.gather(self.queue_lr, self.idx)
            self.queue_gt = tf.gather(self.queue_gt, self.idx)

            # 从数据池中抽取数据
            o_new_lr_imgs = self.queue_lr[0 : new_lr_imgs.shape[0]]
            o_new_hr_imgs = self.queue_gt[0 : new_hr_imgs.shape[0]]

            # 在数据池中加入新数据
            self.queue_lr = tf.concat(
                [self.queue_lr[new_lr_imgs.shape[0] :], new_lr_imgs], axis=0
            )
            self.queue_gt = tf.concat(
                [self.queue_gt[new_hr_imgs.shape[0] :], new_hr_imgs], axis=0
            )
            assert self.queue_gt.shape[0] == self.pool_size

            return (
                o_new_lr_imgs,
                o_new_hr_imgs,
            )
        # 数据池未满
        else:
            self.queue_lr = tf.concat([self.queue_lr, new_lr_imgs], axis=0)
            self.queue_gt = tf.concat([self.queue_gt, new_hr_imgs], axis=0)

            # 由于每次入池的数据量可能小于批次大小，故需要考虑数据池溢出的情况
            if self.queue_gt.shape[0] > self.pool_size:
                self.queue_lr = self.queue_lr[self.queue_lr.shape[0] - self.pool_size :]
                self.queue_gt = self.queue_gt[self.queue_gt.shape[0] - self.pool_size :]
                assert self.queue_gt.shape[0] == self.pool_size

            return new_lr_imgs, new_hr_imgs
