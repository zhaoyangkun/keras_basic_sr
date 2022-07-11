import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from PIL import Image
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.losses import CategoricalCrossentropy, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta, RMSprop
from tensorflow.keras.utils import to_categorical

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

matplotlib.use("TkAgg")

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

print("Compute dtype: %s" % policy.compute_dtype)
print("Variable dtype: %s" % policy.variable_dtype)

batch_size = 16
num_classes = 10

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

"""
将数据集中图片展示出来
"""


def show_mnist(train_image, train_labels):
    n = 3
    m = 3
    fig = plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n, m, i * n + j + 1)
            # plt.subplots_adjust(wspace=0.2, hspace=0.8)
            index = i * n + j  # 当前图片的标号
            img_array = train_image[index]
            img = Image.fromarray(img_array)
            plt.title(train_labels[index])
            plt.imshow(img, cmap="Greys")
    plt.show()


img_row, img_col, channel = 28, 28, 1

mnist_input_shape = (img_row, img_col, 1)

# 将数据维度进行处理
train_images = train_images.reshape(train_images.shape[0], img_row, img_col, channel)
test_images = test_images.reshape(test_images.shape[0], img_row, img_col, channel)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

## 进行归一化处理
train_images /= 255
test_images /= 255

# 将类向量，转化为类矩阵
# 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


"""
构造网络结构
"""
# model = Sequential()
# model.add(
#     Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=mnist_input_shape)
# )
# # kernalsize = 3*3 并没有改变数据维度
# model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # 进行数据降维操作
# model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化，
# # 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
# model.add(Dense(32, activation="relu"))
# # 全连接层
# model.add(Dense(num_classes))
# model.add(Activation("softmax", dtype="float32", name="predictions"))


def build_model():
    input = Input(shape=mnist_input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(input)
    x = Conv2D(16, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(num_classes)(x)
    x = Activation("softmax", dtype="float32", name="predictions")(x)

    return Model(inputs=input, outputs=x)


model = build_model()

"""
编译网络模型,添加一些超参数
"""

# model.compile(
#     loss=categorical_crossentropy,
#     optimizer=Adadelta(),
#     metrics=["accuracy"],
# )

# model.fit(
#     train_images,
#     train_labels,
#     batch_size=batch_size,
#     epochs=5,
#     verbose=1,
#     validation_data=(test_images, test_labels),
#     shuffle=True,
# )


loss_object = CategoricalCrossentropy()
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(len(train_images))
    .batch(batch_size)
)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
    batch_size
)

optimizer = RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def test_step(x):
    return model(x, training=False)


for epoch in range(5):
    epoch_loss_avg = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    for x, y in train_dataset:
        loss = train_step(x, y)
        epoch_loss_avg(loss)
    # for x, y in test_dataset:
    #     predictions = test_step(x)
    #     test_accuracy.update_state(y, predictions)
    print("Epoch {}: loss={}".format(epoch, epoch_loss_avg.result()))

# score = model.evaluate(test_images, test_labels, verbose=1)

# print("test loss:", score[0])
# print("test accuracy:", score[1])
