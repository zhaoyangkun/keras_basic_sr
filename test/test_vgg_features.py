from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model

class Vgg19FeaturesModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def build_vgg_features_model():
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.summary()


build_vgg_features_model()
