import tensorflow as tf


def build_cnn(num_classes: int, image_size: int = 224) -> tf.keras.Model:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_zero_one")(inputs)
    x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(x)
    x = tf.keras.layers.RandomRotation(0.08, name="random_rotation")(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="waste_cnn")

