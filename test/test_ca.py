import sys

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

sys.path.append("./")
from util.layer import channel_attention

input = Input(shape=(64, 64, 32))
# b,h,w,c = input.get_shape().as_list()
# print(b,h,w,c)
output = channel_attention(input)
print(output.shape)
