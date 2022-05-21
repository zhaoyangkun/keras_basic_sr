import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import tensorflow as tf
from tqdm import tqdm


class DataLoader(object):
    """
    数据加载类
    """

    def __init__(
        self,
        train_resource_path,
        test_resource_path,
        batch_size=4,
        hr_img_height=128,
        hr_img_width=128,
        scale_factor=4,
        max_workers=4,
        data_enhancement_factor=1,
    ):
        self.train_resource_path = train_resource_path  # 训练图片资源路径
        self.test_resource_path = test_resource_path  # 测试图片资源路径
        self.batch_size = batch_size  # 单次训练的图片数量
        self.hr_img_height = hr_img_height  # 原图高度
        self.hr_img_width = hr_img_width  # 原图宽度
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
            glob(os.path.join(self.train_resource_path, "*[.png,.jpg,.jpeg,.bmp,.gif]"))
        )
        train_resource_path_list = []
        # 数据增强
        for _ in range(self.data_enhancement_factor):
            train_resource_path_list += ori_train_resource_path_list
        # 处理图片
        train_lr_img_list, train_hr_img_list = self.process_img_data(
            train_resource_path_list, is_center_crop=False
        )

        # 构建训练数据集
        self.train_data = (
            tf.data.Dataset.from_tensor_slices((train_lr_img_list, train_hr_img_list))
            .shuffle(len(train_lr_img_list))  # 打乱数据
            .batch(self.batch_size)  # 批次大小
            .prefetch(tf.data.experimental.AUTOTUNE)  # 预存数据来提升性能
        )

        print(
            "\n", "**********" * 2 + " start creating test dataset " + "**********" * 2
        )
        # 获取所有测试图片路径
        test_resource_path_list = sorted(
            glob(os.path.join(self.test_resource_path, "*[.png,.jpg,.jpeg,.bmp,.gif]"))
        )
        # 处理图片
        test_lr_img_list, test_hr_img_list = self.process_img_data(
            test_resource_path_list,
            is_center_crop=True,
            is_random_crop=False,
            is_random_flip=False,
        )
        # 构建测试数据集
        self.test_data = (
            tf.data.Dataset.from_tensor_slices((test_lr_img_list, test_hr_img_list))
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def process_img_data(
        self,
        resource_path_list,
        is_random_flip=True,
        is_random_crop=True,
        is_center_crop=False,
    ):
        """处理图片数据

        Args:
            resource_path_list (_type_): 图片路径列表
            is_random_flip (bool, optional): 是否随机翻转. Defaults to True.
            is_random_crop (bool, optional): 是否随机裁剪. Defaults to True.
            is_center_crop (bool, optional): 是否中心裁剪. Defaults to False.

        Raises:
            Exception: 图片格式错误

        Returns:
            tf.Tensor, tf.Tensor: 下采样图片列表，原始图片列表
        """
        # 下采样图片列表
        lr_img_list = tf.constant(
            0,
            shape=[
                0,
                self.hr_img_height // self.scale_factor,
                self.hr_img_width // self.scale_factor,
                3,
            ],
            dtype=tf.float32,
        )
        # 原始图片列表
        hr_img_list = tf.constant(
            0, shape=[0, self.hr_img_height, self.hr_img_width, 3], dtype=tf.float32,
        )

        # 多线程处理图片
        pbar = tqdm(total=len(resource_path_list))  # 创建进度条
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for (lr_img, hr_img) in pool.map(
                self.process_img_data_worker,
                resource_path_list,
                [is_random_flip] * len(resource_path_list),
                [is_random_crop] * len(resource_path_list),
                [is_center_crop] * len(resource_path_list),
            ):
                # 扩大维度
                lr_img = tf.expand_dims(lr_img, axis=0)
                hr_img = tf.expand_dims(hr_img, axis=0)
                # 拼接张量
                lr_img_list = tf.concat([lr_img_list, lr_img], axis=0)
                hr_img_list = tf.concat([hr_img_list, hr_img], axis=0)
                # 更新进度条
                pbar.update(1)
            pbar.close()  # 关闭进度条

        return lr_img_list, hr_img_list

    def process_img_data_worker(
        self, path, is_random_flip=True, is_random_crop=True, is_center_crop=False,
    ):
        """
        多线程处理图片工作函数
        """
        # 读取图片
        hr_img = tf.io.read_file(path)

        # 获取图片格式
        img_format = path.split(".")[-1]
        # 根据根据图片格式进行解码
        if img_format == "png":
            hr_img = tf.image.decode_png(hr_img, channels=3)
        elif img_format == "jpg" or img_format == "jpeg":
            hr_img = tf.image.decode_jpeg(hr_img, channels=3)
        elif img_format == "bmp":
            hr_img = tf.image.decode_bmp(hr_img, channels=3)
        elif img_format == "gif":
            hr_img = tf.image.decode_gif(hr_img, channels=3)
        else:
            raise Exception("Unknown image format!")

        # 随机剪裁
        if is_random_crop:
            hr_img = tf.image.random_crop(
                hr_img, [self.hr_img_height, self.hr_img_width, 3]
            )

        # 中心裁剪
        if is_center_crop:
            offset_height = hr_img.shape[0] // 2 - self.hr_img_height // 2
            offset_width = hr_img.shape[1] // 2 - self.hr_img_width // 2
            hr_img = tf.image.crop_to_bounding_box(
                hr_img,
                offset_height,
                offset_width,
                self.hr_img_height,
                self.hr_img_width,
            )

        # 随机水平翻转
        if is_random_flip:
            hr_img = tf.image.random_flip_left_right(hr_img)

        # 归一化
        hr_img = tf.cast(hr_img, tf.float32) / 127.5 - 1
        # hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)

        # 双三次下采样（基于 tensorflow）
        lr_img = tf.image.resize(
            hr_img,
            [
                self.hr_img_height // self.scale_factor,
                self.hr_img_width // self.scale_factor,
            ],
            method=tf.image.ResizeMethod.BICUBIC,
        )

        return lr_img, hr_img
