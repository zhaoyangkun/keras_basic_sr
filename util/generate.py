import sys

import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.append("./")
from util.data_util import denormalize, normalize, resize

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_and_process_image(img_path, normalized_interval=(-1, 1)):
    """读取并处理图片

    Args:
        img_path (str): 图片路径

    Returns:
        Tensor: 图片
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = normalize(image, normalized_interval)
    return image


def generate_different_interpolation_img(img_path_list, scale_factor, save_path):
    """
    生成不同插值方式的图片
    """
    fig, axs = plt.subplots(len(img_path_list), 5)
    for i, img_path in enumerate(img_path_list):
        img = read_and_process_image(img_path=img_path, normalized_interval=(0, 1))
        bicubic_img = resize(
            img,
            img.shape[1] // scale_factor,
            img.shape[0] // scale_factor,
            3,
            mode="bicubic",
        )
        bilinear_img = resize(
            img,
            img.shape[1] // scale_factor,
            img.shape[0] // scale_factor,
            3,
            mode="bilinear",
        )
        area_img = resize(
            img,
            img.shape[1] // scale_factor,
            img.shape[0] // scale_factor,
            3,
            mode="area",
        )
        nearest_img = resize(
            img,
            img.shape[1] // scale_factor,
            img.shape[0] // scale_factor,
            3,
            mode="nearest",
        )

        # 绘制图像
        axs[i, 0].imshow(img)
        axs[i, 0].axis("off")
        if i == 0:
            axs[i, 0].set_title("HR")

        axs[i, 1].imshow(bicubic_img)
        axs[i, 1].axis("off")
        if i == 0:
            axs[i, 1].set_title("Bicubic")

        axs[i, 2].imshow(bilinear_img)
        axs[i, 2].axis("off")
        if i == 0:
            axs[i, 2].set_title("Bilinear")

        axs[i, 3].imshow(area_img)
        axs[i, 3].axis("off")
        if i == 0:
            axs[i, 3].set_title("Area")

        axs[i, 4].imshow(nearest_img)
        axs[i, 4].axis("off")
        if i == 0:
            axs[i, 4].set_title("Nearest")
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=500,
        bbox_inches="tight",
    )
    fig.clear()
    plt.close(fig)


def load_model(model_path):
    """_summary_

    Args:
        model_path (str): 模型路径

    Returns:
        tf.keras.models: 模型
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def add_gaussian_noise(img, mean=0, sigma=1):
    """添加高斯噪声

    Args:
        img (ndarray): 图片
        mean (int, optional): 均值
        sigma (int, optional): 标准差
    """
    img = np.array(img / 255, dtype=np.float32)
    noise = np.random.normal(mean, sigma, img.shape)
    noise_img = img + noise
    noise_img = np.clip(noise_img, 0, 1)
    noise_img = np.uint8(noise_img * 255)

    return noise_img


def generate_sr_img(generator, lr_img_path, sr_img_save_path):
    """生成超分图片

    Args:
        generator (tf.keras.models): 模型路径
        lr_img_path (str): 低分辨率图片路径
        sr_img_save_path (str): 保存图片路径
    """
    lr_img = read_and_process_image(lr_img_path)
    sr_img = generator.predict(tf.expand_dims(lr_img, 0))
    sr_img = tf.squeeze(sr_img, axis=0)
    sr_img = denormalize(sr_img)
    plt.imsave(sr_img_save_path, sr_img.numpy(), format="jpeg")


def generate_sr_img_2(generator, lr_img):
    """生成超分图片

    Args:
        generator (tf.keras.models): 模型
        lr_img (ndarray): 低分辨率图片
    """
    lr_img = lr_img.astype(np.float32)  # uint8 ->  float32
    lr_img = tf.convert_to_tensor(lr_img, dtype=tf.float32)  # ndarray -> tensor
    lr_img = normalize(lr_img, normalized_interval=(-1, 1))  # 归一化
    sr_img = generator.predict(tf.expand_dims(lr_img, 0))  # 生成超分图片
    sr_img = tf.squeeze(sr_img, axis=0)  # 降维
    sr_img = denormalize(sr_img, normalized_interval=(-1, 1))  # 反归一化
    sr_img = sr_img.numpy()  # tensor -> ndarray
    sr_img = sr_img.astype(np.uint8)  # float32 -> uint8

    return sr_img


def generate_area_sr_img(generator_list, img_path, downsample_mode="bicubic"):
    """
    生成矩形区域的超分图片
    """
    # 读取图片
    img = cv.imread(img_path)
    cv.imshow("original", img)

    # 选择 ROI
    x, y, w, h = cv.selectROI(
        windowName="original", img=img, showCrosshair=True, fromCenter=False
    )
    crop = img[y : y + h, x : x + w]

    # 对 ROI 区域进行退化处理
    if downsample_mode == "bicubic":
        crop_lr = cv.resize(
            crop, (crop.shape[1] // 4, crop.shape[0] // 4), interpolation=cv.INTER_CUBIC
        )
    elif downsample_mode == "blur":
        crop_lr = cv.GaussianBlur(crop, (7, 7), 2)
    elif downsample_mode == "noise":
        crop_lr = add_gaussian_noise(crop, mean=0, sigma=0.06)
    else:
        raise ValueError(
            "Unsupported downsample mode! Please set bicubic, blur or noise."
        )

    # 将图片从 BGR 格式转换为 RGB 格式
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    crop_lr = cv.cvtColor(crop_lr, cv.COLOR_BGR2RGB)

    # 生成超分图像
    sr_img_list = []
    for generator in generator_list:
        if generator["resize_factor"] > 1:
            crop_lr_large = cv.resize(
                crop_lr,
                (
                    crop_lr.shape[1] * generator["resize_factor"],
                    crop_lr.shape[0] * generator["resize_factor"],
                ),
                interpolation=cv.INTER_CUBIC,
            )
            sr_img = generate_sr_img_2(generator["model"], crop_lr_large)
        else:
            sr_img = generate_sr_img_2(generator["model"], crop_lr)
        sr_img_list.append(sr_img)

    # 在原图上绘制边框
    left_top = (x, y + h)
    right_bottom = (x + w, y)
    cv.rectangle(img, left_top, right_bottom, (0, 0, 255), 2)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 处理子图
    def make_ticklabels_invisible(fig, title_list=[]):
        for i, ax in enumerate(fig.axes):
            # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.set_title(title_list[i], y=-0.35, fontsize=10)
            ax.axis("off")
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)

    fig = plt.figure(figsize=(6, 2), dpi=300)
    ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    ax1.imshow(img)
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax2.imshow(crop_lr)
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.imshow(sr_img_list[0])
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax4.imshow(sr_img_list[1])
    ax5 = plt.subplot2grid((2, 4), (1, 1))
    ax5.imshow(sr_img_list[2])
    ax6 = plt.subplot2grid((2, 4), (1, 2))
    ax6.imshow(sr_img_list[3])
    ax7 = plt.subplot2grid((2, 4), (1, 3))
    ax7.imshow(sr_img_list[4])

    make_ticklabels_invisible(
        fig=plt.gcf(),
        title_list=[
            "",
            downsample_mode.title(),
            "SRCNN",
            "VDSR",
            "SRGAN",
            "ESRGAN",
            "HA-ESRGAN",
        ],
    )
    plt.tight_layout()
    fig.savefig(
        "./image/{}_contract.png".format(downsample_mode),
        dpi=300,
        bbox_inches="tight",
    )
    fig.clear()
    plt.close(fig)

    k = cv.waitKey(0)
    if k == 27:  # 按 esc 键即可退出
        cv.destroyAllWindows()


if __name__ == "__main__":
    # # 模型路径
    # model_path = "/home/zyk/下载/gen_model_epoch_140"
    # # model_path = (
    # #     "/run/media/zyk/Data/研究生资料/超分模型结果/esrgan/models/train/gen_model_epoch_6500"
    # # )
    # # model_path = (
    # #     "/run/media/zyk/Data/研究生资料/超分模型结果/srgan/models/train/gen_model_epoch_1000"
    # # )
    # # model_path = "/run/media/zyk/Data/研究生资料/超分模型结果/rs-esrgan/gen_model_epoch_200"
    # model_path = "/run/media/zyk-arch/Data/研究生资料/超分模型结果/rs-esrgan-bicubic/models/train/gen_model_epoch_1000"
    # # 加载模型
    # generator = load_model(model_path)
    # # 获取图片路径
    # lr_img_resource_path_list = sorted(
    #     glob(os.path.join(
    #         "./image/set14/LRbicx4",
    #         "*[.png]",
    #     )))
    # # 生成图片
    # for lr_img_path in lr_img_resource_path_list:
    #     sr_img_save_path = lr_img_path.replace("LRbicx4", "rs-esrgan-bicubic")
    #     generate_sr_img(generator, lr_img_path, sr_img_save_path)

    # 加载模型，选择 ROI 区域进行超分，生成超分效果对比图
    srcnn = load_model("F:/研究生资料/超分模型结果/srcnn/models/train/gen_model_epoch_200")
    vdsr = load_model("F:/研究生资料/超分模型结果/vdsr/models/train/gen_model_epoch_200")
    srgan = load_model("F:/研究生资料/超分模型结果/srgan/models/train/gen_model_epoch_100")
    esrgan = load_model("F:/研究生资料/超分模型结果/esrgan/models/train/gen_model_epoch_100")
    ha_esrgan = load_model("F:/研究生资料/超分模型结果/ha-esrgan/models/train/gen_model_epoch_100")
    generate_area_sr_img(
        [
            {"model": srcnn, "resize_factor": 4},
            {"model": vdsr, "resize_factor": 4},
            {"model": srgan, "resize_factor": 1},
            {"model": esrgan, "resize_factor": 1},
            {"model": ha_esrgan, "resize_factor": 1},
        ],
        "./image/head.jpg",
        downsample_mode="bicubic",
    )
