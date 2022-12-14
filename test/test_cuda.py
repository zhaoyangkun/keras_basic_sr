import tensorflow as tf
import torch

# print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices("GPU"))

print(torch.cuda.is_available())
