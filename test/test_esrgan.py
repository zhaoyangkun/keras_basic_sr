import os
import sys

import tensorflow as tf

sys.path.append("./")
from util.generate import generate_sr_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = "F:/研究生资料/超分模型结果/rs-esrgan/gen_model_epoch_200"
lr_img_path = "F:\\projects\\python\\tensorflow\\keras_basic_sr\\image\\hehua\\03.jpg"
sr_img_save_path = "./03_sr_rs.jpg"

generator = tf.keras.models.load_model(
    model_path,
    compile=False,
)

generate_sr_img(generator=generator,
                lr_img_path=lr_img_path,
                sr_img_save_path=sr_img_save_path)
