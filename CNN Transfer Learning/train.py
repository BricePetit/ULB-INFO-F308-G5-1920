import os

import numpy as np
import keras

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


from keras.models import Sequential

def train(train_generator):
	
	base_model = VGG19(weights='imagenet',include_top=False)

	for layer in base_model.layers[:len(base_model.layers)]:
		layer.trainable = False

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(256,activation='relu')(x)
	out = Dense(5,activation='softmax')(x)

	model = Model(inputs=base_model.input,outputs=out)

	model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

	step_size_train = train_generator.n//train_generator.batch_size


	model.fit_generator(generator=train_generator,
						steps_per_epoch=step_size_train,
						#epochs=5,
						epochs=10,
						verbose=1)
	return model


def crossValidation(nb_itr=1, validation_split=0.2):
	acc = 0

	for i in range(nb_itr):

		train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=validation_split)

		#folder = "./data_tiny/"
		folder = "./data/"
		#folder = "./data_mid/"

		train_generator = train_datagen.flow_from_directory(folder,
															target_size=(224,224),
															color_mode='rgb',
															batch_size=32,
															class_mode='categorical',
															shuffle=True,
															subset='training')

		validation_generator = train_datagen.flow_from_directory(folder,
																target_size=(224,224),
																color_mode='rgb',
																batch_size=32,
																class_mode='categorical',
																shuffle=False,
																subset='validation')


		model = train(train_generator)

		#current_acc = model.evaluate(validation_generator,verbose=1)[1]

		result = model.predict(validation_generator)
		validation_generator.reset()

		nb_correct = 0
		confusion_matrix = np.array([[0 for _ in range(5)] for _ in range(5)])

		for x in range(len(validation_generator.classes)):
			if np.argmax(result[x]) == validation_generator.classes[x]:
				nb_correct += 1

			confusion_matrix[validation_generator.classes[x]][np.argmax(result[x])] += 1

		current_acc = nb_correct / len(validation_generator.classes)


		print("Accuracy of iteration",i+1,":",round(current_acc*100,2),"%")
		acc += current_acc
		show_confusion_matrix(confusion_matrix,current_acc,i+1)

	print("Average accuracy :",round((acc/nb_itr)*100,2),"%")

def show_confusion_matrix(confusion_matrix,accuracy,it):
	categories = ["Blanc","Bleu","Jaune","Orange","Verre"]

	n = len(categories)
	fig, ax = plt.subplots()
	im = ax.imshow(confusion_matrix)

	ax.set_xticks(np.arange(n))
	ax.set_yticks(np.arange(n))

	ax.set_xticklabels(categories)
	ax.set_yticklabels(categories)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

	for i in range(n):
		for j in range(n):
			text = ax.text(j, i, confusion_matrix[i, j],ha="center", va="center", color="w")

	ax.set_title("Prediction accuracy: {0}%".format(round(accuracy*100,2)))
	fig.tight_layout()
	fig.savefig("confusion_matrix/fig"+str(it)+".png")


if __name__ == '__main__':
	crossValidation(5,0.2)
