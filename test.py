import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report 

model = tf.keras.models.load_model('trained_model.keras')
model_sgd = tf.keras.models.load_model('sgd_with_augmentation_and_normalization.keras')  

print('ADAM')
model.summary()
print('-------------------------------------')
print('SGD')
model_sgd.summary()

test_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred", #labels are selected based on folder names
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
)

class_name = test_set.class_names

y_pred = model.predict(test_set)
y_pred_sgd = model_sgd.predict(test_set)

predicted_categories = tf.argmax(y_pred, axis=1)
predicted_categories_sgd = tf.argmax(y_pred_sgd, axis=1)

true_categories = tf.concat([y for x,y in test_set], axis=0)

Y_true = tf.argmax(true_categories, axis=1)


print(classification_report(Y_true, predicted_categories, target_names = class_name))
print('-------------------------------------')
print(classification_report(Y_true, predicted_categories_sgd, target_names = class_name))

image_path = "test\TomatoYellowCurlVirus4.JPG"

img = cv2.imread(image_path)

image = tf.keras.preprocessing.image.load_img(image_path, target_size = (128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

prediction = model.predict(input_arr)
prediction_sgd = model_sgd.predict(input_arr)

result_index = np.argmax(prediction)
result_index_sgd = np.argmax(prediction_sgd) 

model_prediction = class_name[result_index]
model_sgd_prediction = class_name[result_index_sgd]

print('ADAM prediction: ' + model_prediction)

print('SGD prediction: ' + model_sgd_prediction)