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

	model = Sequential()
	model.add(base_model)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(412,activation='relu'))
	model.add(Dense(312,activation='relu'))
	model.add(Dense(212,activation='relu'))
	model.add(Dense(112,activation='relu'))
	model.add(Dense(12,activation='relu'))
	#model.add(Dense(5*4,activation='relu'))
	model.add(Dense(5,activation='softmax'))

	#model.summary()

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
		test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=validation_split)

		#folder = "data_tiny"
		#folder = "data"
		folder = "data_mid"
		
		train_generator,test_generator = split(train_datagen,test_datagen,folder,0.2)

		model = train(train_generator)

		#current_acc = model.evaluate(test_generator,verbose=1)[1]

		result = model.predict(test_generator)

		nb_correct = 0
		confusion_matrix = np.array([[0 for _ in range(5)] for _ in range(5)])

		for x in range(len(test_generator.classes)):
			if np.argmax(result[x]) == test_generator.classes[x]:
				nb_correct += 1

			confusion_matrix[test_generator.classes[x]][np.argmax(result[x])] += 1

		current_acc = nb_correct / len(test_generator.classes)
		show_confusion_matrix(confusion_matrix,current_acc,i+1)


		print("Accuracy of iteration",i+1,":",round(current_acc*100,2),"%")
		acc += current_acc

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


def find_best_hyperparameters():
	pass

import os
import random
import shutil

def split(train_datagen,test_datagen,directory,test_split=0.2):
	try:
		shutil.rmtree("datasets")
	except:
		pass

	os.mkdir("datasets")
	os.mkdir("datasets/train")
	os.mkdir("datasets/test")

	for sub_dir in os.listdir(directory):
		images = os.listdir(directory + "/" + sub_dir)
		random.shuffle(images)
		test_images = images[:round(test_split*len(images))]
		train_images = images[round(test_split*len(images)):]

		os.mkdir("datasets/test/" + sub_dir)

		for i in test_images:
			os.link(directory + "/" + sub_dir + "/" + i,"datasets/test/" + sub_dir + "/" + i)

		os.mkdir("datasets/train/" + sub_dir)

		for i in train_images:
			os.link(directory + "/" + sub_dir + "/" + i,"datasets/train/" + sub_dir + "/" + i)

	train_generator = train_datagen.flow_from_directory("datasets/train/",
													target_size=(224,224),
													color_mode='rgb',
													batch_size=32,
													class_mode='categorical')

	test_generator = test_datagen.flow_from_directory("datasets/test/",
													target_size=(224,224),
													color_mode='rgb',
													batch_size=32,
													shuffle=False,
													class_mode='categorical')

	return train_generator,test_generator

if __name__ == '__main__':
	crossValidation(5,0.2)
