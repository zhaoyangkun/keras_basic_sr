import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.append("./")
from util.data_util import resize

# 缩放倍数
scale_factor = 4

# 原图目录
hr_img_dir = "./image/set5/GTmod12"

# 下采样图像保存目录
lr_img_dir = Path("./image/set5") / "LRbicx{}".format(scale_factor)

# 创建下采样图像保存目录
lr_img_dir.mkdir(parents=True, exist_ok=True)

# 获取原图目录下的所有图片
hr_img_path_list = [
    str(file) for file in Path(hr_img_dir).rglob("*[.png, .jpg, .jpeg, .bmp]")
]

for hr_img_path in hr_img_path_list:
    print("Processing: {}".format(hr_img_path))
    # 读取图片
    hr_img = cv.imdecode(np.fromfile(hr_img_path, dtype=np.uint8), cv.IMREAD_COLOR)
    # 将图片从 BGR 格式转换为 RGB 格式
    hr_img = cv.cvtColor(hr_img, cv.COLOR_BGR2RGB)
    # np.unit8 --> tf.uint8
    hr_img = tf.convert_to_tensor(hr_img, dtype=tf.uint8)
    # 双三次插值下采样
    lr_img = resize(
        hr_img,
        hr_img.shape[1] // scale_factor,
        hr_img.shape[0] // scale_factor,
        3,
        mode="bicubic",
    )
    lr_img_path = lr_img_dir / Path(hr_img_path).name
    plt.imsave(lr_img_path, lr_img.numpy())
