import cv2 as cv
import numpy as np


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


img = cv.imread("./image/RealSR_JPEG/comic3.png")
noise_img = add_gaussian_noise(img, mean=0, sigma=10)
cv.imshow("out", noise_img)
cv.waitKey()
