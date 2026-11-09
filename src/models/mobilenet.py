import tensorflow as tf


def build_mobilenet(
    num_classes: int,
    image_size: int = 224,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocessing"
    )(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="waste_mobilenet_v2")

