import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19


def create_vgg19_custom_model():
    # Block 1
    input = tf.keras.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, (3, 3), padding="same", name="block1_conv1")(input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="block1_conv2")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="block2_conv2")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv3")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv4")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding="same", name="block4_conv1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block4_conv2")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block4_conv3")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block4_conv4")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), padding="same", name="block5_conv1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block5_conv2")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block5_conv3")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="block5_conv4")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    return Model(input, x, name="vgg19_features")


# class Vgg19FeaturesModel(Model):
#     def __init__(self):
#         super(Vgg19FeaturesModel, self).__init__()
#         conv1 =

#     def call(self, inputs):
#         pass


class Vgg19FeaturesModel(tf.keras.Model):
    def __init__(self):
        super(Vgg19FeaturesModel, self).__init__()
        original_vgg = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet"
        )
        self.vgg_model = create_vgg19_custom_model()
        self.vgg_model.set_weights(original_vgg.get_weights())

        layers = [
            "block1_conv2",
            "block2_conv2",
            "block3_conv4",
            "block4_conv4",
            "block5_conv4",
        ]
        outputs = [self.vgg_model.get_layer(name).output for name in layers]

        self.model = tf.keras.Model([self.vgg_model.input], outputs)
        self.model.trainable = False

    def call(self, x):
        x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
        return self.model(x)
