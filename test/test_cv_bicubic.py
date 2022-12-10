import cv2 as cv
import tensorflow as tf

img_1 = cv.imread("./image/set5/original/baby.png")
img_1 = tf.convert_to_tensor(img_1, dtype=tf.float32)
print(img_1)
img_1 = tf.clip_by_value(img_1, -1, 1)
print(img_1)
img_1 = tf.clip_by_value(img_1, 0, 255)


# img_1_last = tf.cast((img_1 + 1) * 127.5, dtype=tf.uint8)
# print(img_1)

# img_2 = cv.imread("./image/set5/original/bird.png")

# img_1 = cv.resize(img_1, (128, 128), interpolation=cv.INTER_CUBIC)
# img_2 = cv.resize(img_2, (128, 128), interpolation=cv.INTER_CUBIC)

# img_1 = np.expand_dims(img_1, axis=0)
# img_2 = np.expand_dims(img_2, axis=0)

# ori_imgs = np.concatenate([img_1, img_2], axis=0)
# proc_imgs = np.empty([0, 256, 256, 3], dtype=np.uint8)

# for i in range(ori_imgs.shape[0]):
#     proc_img = cv.resize(ori_imgs[i], (256, 256), interpolation=cv.INTER_CUBIC)
#     proc_imgs = np.concatenate([proc_imgs, np.expand_dims(proc_img, axis=0)], axis=0)

# print(proc_imgs)
# print(img_list.ndim)
# img_list = cv.resize(img_list, (256, 256), interpolation=cv.INTER_CUBIC)
# # img_1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # print(np.max(img), np.min(img))
# # img = tf.image.random_crop(img, [128, 128, 3])
# # print(img.shape)
# plt.imshow(img.numpy())
# plt.show()

# import cv2


# mode_dict = {
#     "area": cv2.INTER_AREA,
#     "bilinear": cv2.INTER_LINEAR,
#     "bicubic": cv2.INTER_CUBIC,
# }

# print("area" in mode_dict)
# print("bilinear" in mode_dict)
# print("bicubic" in mode_dict)
# print("bicubi" in mode_dict)
