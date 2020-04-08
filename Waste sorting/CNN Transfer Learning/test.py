import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# load model
# model = load_model('test.model')

categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

# load the model
# model = VGG16()
# model = load_model('test.model')
model = load_model('Model.model')
# model.summary()
image_path = "yg.jpg"
img = image.load_img(image_path, target_size=(224, 224))
plt.imshow(img)
img = np.expand_dims(img, axis=0)
result = model.predict([img])
# plt.title(get_label_name(result[0][0]))
# label = ["Blanc","Bleu","Jaune"]
# print(result)
plt.title(categories[np.argmax(result[0])])
plt.show()

# load an image from file
# image = load_img('1.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array
# image = img_to_array(image)

# reshape data for the model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model
# image = preprocess_input(image)

# predict the probability across all output classes
# yhat = model.predict(image)

# convert the probabilities to class labels
# label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability
# label = label[0][0]

# print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))
