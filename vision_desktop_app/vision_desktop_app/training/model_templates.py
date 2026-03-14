import tensorflow as tf
from tensorflow.keras import layers, models


def classification_model(input_shape=(128, 128, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def detection_model_simple(input_shape=(128, 128, 3), num_classes=2):
    """
    Simple single-bbox detector: returns class probabilities and bounding box (normalized x,y,w,h)
    Loss: classification + box regression (MSE)
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)

    class_out = layers.Dense(num_classes, activation='softmax', name='class_out')(x)
    bbox_out = layers.Dense(4, activation='sigmoid', name='bbox_out')(x)  # normalized to 0..1

    model = models.Model(inputs, [class_out, bbox_out])
    # custom losses will be applied in trainer
    model.compile(optimizer='adam', loss={'class_out': 'sparse_categorical_crossentropy', 'bbox_out': 'mse'}, metrics={'class_out': 'accuracy'})
    return model
