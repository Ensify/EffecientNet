import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2B3
from utils import plot_metrics

# Define paths and parameters
dataset_path = '/content/defect_classification'
batch_size = 32
img_height = 384
img_width = 384
validation_split = 0.2
num_classes = 2
epochs = 1

# Create a MirroredStrategy for distributed training
strategy = tf.distribute.MirroredStrategy()

# Load dataset and split it into training and validation sets
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

with strategy.scope():
    # Load EfficientNetB0 pre-trained on ImageNet without the top layer
    model = EfficientNetV2B3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=num_classes,
        classifier_activation="softmax",
        include_preprocessing=True,
    )

    # Freeze the base model
    model.trainable = True

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    verbose=1
)

plot_metrics(history)

# Save checkpoint
model.save('model_efficientnetb3.keras')
