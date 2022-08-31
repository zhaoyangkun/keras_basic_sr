import sys
from tensorflow.keras import backend as K
import tensorflow as tf

sys.path.append("./")
from util.metric import calculate_psnr

a = tf.constant([1., 2., 3.], dtype=tf.float32)
b = tf.constant([4., 5., 6.], dtype=tf.float32)

a_1 = a / 127.5 - 1
b_1 = b / 127.5 - 1

a_2 = a / 255.0
b_2 = b / 255.0

print((a_1 + 1) / 2)
print(a_2)
print((b_1 + 1) / 2)
print(b_2)


# 32.56778
# metric = calculate_psnr(a, b)
# print(metric)
