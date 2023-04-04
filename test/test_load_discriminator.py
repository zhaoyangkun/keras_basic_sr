import sys

import cv2 as cv
import tensorflow as tf

sys.path.append("./")

from util.data_util import denormalize, load_model, read_and_process_image

discriminator = load_model(
    "F:/研究生资料/超分模型结果/ha-esrgan/models/train/dis_model_epoch_100")

input = read_and_process_image("./image/image1_gray.png")

output = discriminator.predict(tf.expand_dims(input, 0))
output = tf.squeeze(output, axis=0)
output = denormalize(output)
output = output.numpy()
output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

cv.imwrite("./disc_result.png", output)

