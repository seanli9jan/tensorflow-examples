# Import packages
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import os

from utility import utils
utils.tf_disable_logging("WARNING")
utils.tf_limit_gpu()

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Load data
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")
CLASS_NAMES = ["cats", "dogs"]

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

train_cats_dir = os.path.join(train_dir, "cats")  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, "dogs")  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, "cats")  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, "dogs")  # directory with our validation dog pictures

# Understand the data
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("Total training cat images:", num_cats_tr)
print("Total training dog images:", num_dogs_tr)

print("Total validation cat images:", num_cats_val)
print("Total validation dog images:", num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Parameters
BATCH_SIZE = 128
EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

dataloader_type = "tensorflow"

print("DataLoader:", dataloader_type)

# TensorFlow Dataset API
if dataloader_type == "tensorflow":
    train_list_ds = tf.data.Dataset.list_files(train_dir+"/*/*")
    val_list_ds = tf.data.Dataset.list_files(validation_dir+"/*/*")

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES[1]

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    # Data preprocessing
    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Basic methods for training and validation
    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, is_training=True):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if is_training:
            # Data augmentation
            ds = ds.map(
                lambda img, label: (tf.image.random_flip_left_right(img), label)
            )
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            ds = ds.batch(BATCH_SIZE)
            # Repeat forever
            ds = ds.repeat()
        else:
            ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = prepare_for_training(train_labeled_ds, cache="./cache/cats_and_dogs_train.tfcache", shuffle_buffer_size=total_train)
    val_ds = prepare_for_training(val_labeled_ds, cache="./cache/cats_and_dogs_val.tfcache", is_training=False)

# Keras Dataset API
elif dataloader_type == "keras":
    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

    train_ds = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode="binary")
    val_ds = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                            directory=validation_dir,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode="binary")

pretrained = False

print("Transfer learning:", pretrained)

# Optimizer
opt = tf.keras.optimizers.Nadam()

# Custom loss
def sigmoid_crossentropy(y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    first_log = y_true * K.log(y_pred + K.epsilon())
    second_log = (1.0 - y_true) * K.log(1.0 - y_pred + K.epsilon())
    return K.mean(-(first_log + second_log), axis=-1)

#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True),
loss_fn = sigmoid_crossentropy

# Custom metrics
def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels
    Returns:
    Specificity score
    """
    y_pred = K.sigmoid(y_pred)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

custom_objects = {specificity.__name__: specificity}

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
if pretrained:
    # Load model
    model = utils.tf_load_model("checkpoint", custom_objects=custom_objects, compile=False)
    K.clear_session()

    # Change last layer
    name=model.layers[-1].name
    x = model.layers[-2].output
    outputs = Dense(1, name=name)(x)
    model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

else:
    model_type = "Class"

    if model_type == "Sequential":
        # Create sequential model
        model = Sequential([
            Conv2D(16, 3, padding="same", activation="relu", input_shape=input_shape),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(64, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(1)
        ])

    elif model_type == "Functional":
        # Create functional model
        inputs = Input(shape=input_shape)
        x = Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = MaxPooling2D()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, 3, padding="same", activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Conv2D(64, 3, padding="same", activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.2)(x)
        #x = utils.tf_print_tensor(x, message="sum(dropout_1):", map_fn=K.sum)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        outputs = Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif model_type == "Class":
        # Create class model
        class Model(tf.keras.Model):
            def __init__(self, input_shape):
                super(Model, self).__init__()
                self.conv2d_1 = Conv2D(16, 3, padding="same", activation="relu")
                self.max_pooling2d_1 = MaxPooling2D()
                self.dropout_1 = Dropout(0.2)
                self.conv2d_2 = Conv2D(32, 3, padding="same", activation="relu")
                self.max_pooling2d_2 = MaxPooling2D()
                self.conv2d_3 = Conv2D(64, 3, padding="same", activation="relu")
                self.max_pooling2d_3 = MaxPooling2D()
                self.dropout_2 = Dropout(0.2)
                self.flatten_1 = Flatten()
                self.dense_1 = Dense(512, activation="relu")
                self.dense_2 = Dense(1)

                # Add input layer
                self.inputs = Input(shape=input_shape)
                # Get output layer with `call` method
                self.outputs = self.call(self.inputs)

                # Reinitial
                super(Model, self).__init__(
                    inputs=self.inputs,
                    outputs=self.outputs,
                    name=self.name
                )

            def call(self, inputs, training=False):
                x = self.conv2d_1(inputs)
                x = self.max_pooling2d_1(x)
                x = self.dropout_1(x, training=training)
                x = self.conv2d_2(x)
                x = self.max_pooling2d_2(x)
                x = self.conv2d_3(x)
                x = self.max_pooling2d_3(x)
                x = self.dropout_2(x, training=training)
                #x = utils.tf_print_tensor(x, message="sum(dropout_1):", map_fn=K.sum)
                x = self.flatten_1(x)
                x = self.dense_1(x)
                return self.dense_2(x)

        model = Model(input_shape=input_shape)

# Compile the model
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=["accuracy", custom_objects])

# Model summary
model.summary()

# Train the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
history = model.fit(
    train_ds,
    steps_per_epoch=(total_train // BATCH_SIZE) +
                    (total_train % BATCH_SIZE != 0),
    epochs=EPOCHS,
    callbacks=[tensorboard_callback],
    validation_data=val_ds,
)

# Save model
utils.tf_save_model(model, "checkpoint")

print("Run the command line:")
print("--> tensorboard --logdir=./logs --bind_all")
print("Then open localhost:6006 into your web browser")

# Get result
#acc = history.history["accuracy"]
#val_acc = history.history["val_accuracy"]

#loss = history.history["loss"]
#val_loss = history.history["val_loss"]
