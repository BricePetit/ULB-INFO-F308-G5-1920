from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def predict(model, img_path):
	categories = ["Blanc","Bleu","Jaune","Orange","Verre"]
	img = image.load_img(img_path, target_size=(224, 224))
	plt.imshow(img)
	img = np.expand_dims(img, axis=0)
	result = model.predict([img])
	return categories[np.argmax(result[0])]
	#plt.title(categories[np.argmax(result[0])])
	#plt.axis('off')
	#plt.show()

if __name__ == '__main__':

	model = load_model('Model.model')
	#model.summary()

	print(predict(model,"test.jpg"))
