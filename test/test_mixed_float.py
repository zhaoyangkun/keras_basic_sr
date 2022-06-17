import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

inputs = layers.Input(shape=(784,), name="digits")
if tf.config.list_physical_devices("GPU"):
    print("The model will run with 4096 units on a GPU")
    num_units = 4096
else:
    # Use fewer units on CPUs so the model finishes in a reasonable amount of time
    print("The model will run with 64 units on a CPU")
    num_units = 64

dense1 = layers.Dense(num_units, activation="relu", name="dense_1")
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation="relu", name="dense_2")
x = dense2(x)

print(dense1.dtype_policy)
print("x.dtype: %s" % x.dtype.name)
# 'kernel' is dense1's variable
print("dense1.kernel.dtype: %s" % dense1.kernel.dtype.name)

# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
print("Outputs dtype: %s" % outputs.dtype.name)
