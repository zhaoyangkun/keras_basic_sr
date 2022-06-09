import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

patch_height = 400
patch_width = 400

hr_img = tf.io.read_file("./image/test_hr.png")
hr_img = tf.image.decode_png(hr_img, channels=3)

offset_height = hr_img.shape[0] // 2 - patch_height // 2
offset_width = hr_img.shape[1] // 2 - patch_width // 2
hr_img = tf.image.crop_to_bounding_box(
    hr_img, offset_height, offset_width, patch_height, patch_width
)

hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)
hr_img = tf.image.resize(
    hr_img,
    [patch_height // 4, patch_width // 4],
    method=tf.image.ResizeMethod.BICUBIC,
)
hr_img = hr_img.numpy()

# lr_img = cv.resize(hr_img, (patch_height // 4, patch_width // 4), interpolation=cv.INTER_CUBIC)

# print(tf.reduce_max(hr_img), tf.reduce_min(hr_img))

# plt.imshow(hr_img)
# plt.show()
plt.imsave("./image/test_lr_1.png", hr_img)
