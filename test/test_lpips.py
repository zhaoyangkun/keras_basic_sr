import sys
import tensorflow as tf

sys.path.append("./")
from util.metric import calculate_lpips

# 读取图片
lr_img = tf.io.read_file("./image/ex_p0.png")
sr_img = tf.io.read_file("./image/ex_p1.png")
hr_img = tf.io.read_file("./image/ex_ref.png")

# 解码图片
lr_img = tf.image.decode_png(lr_img, channels=3)
sr_img = tf.image.decode_png(sr_img, channels=3)
hr_img = tf.image.decode_png(hr_img, channels=3)

lr_img = tf.expand_dims(lr_img, axis=0)
sr_img = tf.expand_dims(sr_img, axis=0)
hr_img = tf.expand_dims(hr_img, axis=0)

y_true = tf.concat([hr_img, hr_img], axis=0)
y_pred = tf.concat([lr_img, sr_img], axis=0)

metric = calculate_lpips(y_true, y_pred)

print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"lpips metric shape: {metric.shape}")
print(f"ref <-> p0: {metric[0]:.3f}")
print(f"ref <-> p1: {metric[1]:.3f}")
print(f"mean lpips metric: {tf.reduce_mean(metric)}")
