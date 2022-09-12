from glob import glob
import os
import tensorflow as tf

# 加载并处理图片
def load_and_preprocess_image(lr_img_path, hr_img_path):
    print(lr_img_path, hr_img_path)
    # image = tf.io.read_file(path)
    # image = tf.image.decode_png(image, channels=3)
    # image = tf.cast(image, dtype=tf.float32)
    # image = image / 127.5 - 1  # 归一化到 [-1, 1]
    return lr_img_path, hr_img_path


# 获取所有测试图片路径
lr_img_resource_path_list = sorted(
    glob(
        os.path.join(
            "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5/LRbicx4",
            "*[.png]",
        )
    )
)
hr_img_resource_path_list = sorted(
    glob(
        os.path.join(
            "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set5/GTmod12",
            "*[.png]",
        )
    )
)

# 反归一化
def denormalize(image):
    return tf.cast((image + 1) * 127.5, dtype=tf.uint8)


test_data = (
    tf.data.Dataset.from_tensor_slices(
        (lr_img_resource_path_list, hr_img_resource_path_list)
    )
    .map(
        lambda lr_img_path, hr_img_path: load_and_preprocess_image(
            lr_img_path, hr_img_path
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .batch(1)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
