from glob import glob
import os
import tensorflow as tf

# path = "/home/zyk/dataset/test_01.png"
# img_format = tf.strings.split(path, ".")[-1]
# if (img_format == "png"):
#     print(True)


# class Test:
#     def __init__(self):
#         path_list = sorted(
#             glob(
#                 os.path.join(
#                     "/home/zyk/projects/python/tensorflow/keras_basic_sr/image",
#                     "*[.png,.jpg,.jpeg,.bmp,.gif]",
#                 )
#             )
#         )

#         dataset = (
#             tf.data.Dataset.from_tensor_slices((path_list))
#             .map(
#                 lambda image_path: self.parse_path_function(
#                     image_path, 128, 128, True, True, True, False, "train"
#                 )
#             )
#             .batch(1)
#         )

#         for data in dataset:
#             print(data)

#     def parse_path_function(
#         self,
#         img_path,
#         hr_img_height,
#         hr_img_width,
#         is_random_flip=True,
#         is_random_crop=True,
#         is_random_rot=True,
#         is_center_crop=False,
#         mode="train",
#     ):
#         # print(img_path)
#         img = tf.io.read_file(img_path)
#         img_data = tf.image.decode_png(img, 3)

#         return img_data


# test = Test()

# a = tf.constant(2.0)
b = tf.constant(6)
print(b // 2)