import numpy as np
import tensorflow as tf

tf.keras.applications.MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)

trainDir = "dataset/final/train"
train_dataset = tf.keras.utils.image_dataset_from_directory(
    trainDir, shuffle=True, batch_size=32, image_size=(224, 224)
)

validationDir = "dataset/mobilenetv3/valid"
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validationDir, shuffle=True, batch_size=32
)
