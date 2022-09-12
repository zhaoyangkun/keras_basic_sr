
import matplotlib.pyplot as plt
import tensorflow as tf

img = tf.io.read_file("./image/test_1.png")
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.random_crop(img, [32, 32, 3])
img = img.numpy()
# plt.imshow(img)
# plt.show()
plt.imsave("./image/test_lr_1.png", img)
