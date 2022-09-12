import os

from glob import glob
import tensorflow as tf
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_and_process_image(img_path):
    """读取并处理图片

    Args:
        img_path (str): 图片路径

    Returns:
        Tensor: 图片
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 127.5 - 1  # 归一化到 [-1, 1]
    return image


def denormalize(image):
    """反归一化

    Args:
        image (Tensor): 图像

    Returns:
        Tensor: 归一化后的图像
    """
    return tf.cast((image + 1) * 127.5, dtype=tf.uint8)


def load_model(model_path):
    """_summary_

    Args:
        model_path (str): 模型路径

    Returns:
        tf.keras.models: 模型
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def generate_sr_img(generator, lr_img_path, sr_img_save_path):
    """生成超分图片

    Args:
        model (tf.keras.models): 模型路径
        lr_img_path (str): 低分辨率图片路径
        sr_img_save_path (str): 保存图片路径
    """
    lr_img = read_and_process_image(lr_img_path)
    sr_img = generator.predict(tf.expand_dims(lr_img, 0))
    sr_img = tf.squeeze(sr_img, axis=0)
    sr_img = denormalize(sr_img)
    plt.imsave(sr_img_save_path, sr_img.numpy(), format="jpeg")


if __name__ == "__main__":
    # 模型路径
    model_path = "/home/zyk/下载/gen_model_epoch_140"
    # model_path = (
    #     "/run/media/zyk/Data/研究生资料/超分模型结果/esrgan/models/train/gen_model_epoch_6500"
    # )
    # model_path = (
    #     "/run/media/zyk/Data/研究生资料/超分模型结果/srgan/models/train/gen_model_epoch_1000"
    # )
    # 加载模型
    generator = load_model(model_path)
    # 获取图片路径
    lr_img_resource_path_list = sorted(
        glob(
            os.path.join(
                "/home/zyk/projects/python/tensorflow/keras_basic_sr/image/set14/LRbicx4",
                "*[.png]",
            )
        )
    )
    # 生成图片
    for lr_img_path in lr_img_resource_path_list:
        sr_img_save_path = lr_img_path.replace("LRbicx4", "real-esrgan")
        generate_sr_img(generator, lr_img_path, sr_img_save_path)
