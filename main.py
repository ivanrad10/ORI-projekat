import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
###data preprocessing

#train data
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred", #labels are selected based on folder names
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
)

#validation data
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred", #labels are selected based on folder names
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = [128, 128, 3]))
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units = 1500, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.4))

#Output layer
model.add(tf.keras.layers.Dense(units = 38, activation = 'softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

#model training
training_history = model.fit(x = training_set, validation_data = validation_set, epochs = 10)

train_loss, train_accuracy = model.evaluate(training_set)
validation_loss, validation_accuracy = model.evaluate(validation_set)

model.save("trained_model.keras")
