import pytest
import tensorflow as tf

from src.inference.predict import make_feature_extractor
from src.models.cnn import build_cnn
from src.models.mobilenet import build_mobilenet


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    (
        (build_cnn, {}),
        (build_mobilenet, {"weights": None}),
    ),
)
def test_model_output_shape_and_probability_sum(builder, kwargs) -> None:
    model = builder(num_classes=3, image_size=64, **kwargs)
    prediction = model(tf.zeros((2, 64, 64, 3)), training=False)
    assert tuple(prediction.shape) == (2, 3)
    tf.debugging.assert_near(
        tf.reduce_sum(prediction, axis=1), tf.ones((2,)), atol=1e-5
    )
    if builder is build_mobilenet:
        extractor = make_feature_extractor(model)
        features = extractor(tf.zeros((2, 64, 64, 3)), training=False)
        assert features.shape[0] == 2
        assert features.shape[1] > 0


def test_feature_extractor_supports_nested_finetuned_mobilenet() -> None:
    base = build_mobilenet(num_classes=3, image_size=64, weights=None)
    inputs = tf.keras.Input((64, 64, 3))
    augmented = tf.keras.layers.RandomFlip("horizontal")(inputs)
    nested = tf.keras.Model(inputs, base(augmented), name="fine_tuned_wrapper")

    extractor = make_feature_extractor(nested)
    features = extractor(tf.zeros((2, 64, 64, 3)), training=False)

    assert tuple(features.shape) == (2, 1280)
