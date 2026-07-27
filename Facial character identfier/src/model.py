import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model


def build_model(img_size, num_attrs, num_landmarks, num_bbox):
    """Same architecture as the original notebook (cell 34) — untouched."""
    inputs = Input(shape=(img_size, img_size, 3))

    backbone = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    backbone.trainable = False

    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    attributes_out = Dense(num_attrs, activation="sigmoid", name="attributes")(x)
    landmarks_out = Dense(num_landmarks, activation="sigmoid", name="landmarks")(x)
    bbox_out = Dense(num_bbox, activation="sigmoid", name="bbox")(x)

    model = Model(inputs=inputs, outputs=[attributes_out, landmarks_out, bbox_out])
    return model, backbone


def compile_model(model, learning_rate=None):
    """Same loss/metrics as cells 35 and 37. Pass learning_rate=1e-5 for the fine-tune stage."""
    optimizer = "adam" if learning_rate is None else tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={"attributes": "binary_crossentropy", "landmarks": "mse", "bbox": "mse"},
        loss_weights={"attributes": 1.0, "landmarks": 5.0, "bbox": 5.0},
        metrics={"attributes": "accuracy", "landmarks": "mae", "bbox": "mae"},
    )
    return model