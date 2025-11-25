import tensorflow as tf

print(tf.config.list_physical_devices)
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(is_cuda_gpu_available)
