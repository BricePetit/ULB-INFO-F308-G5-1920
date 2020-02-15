import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


def train(train_generator):
	base_model=VGG19(weights='imagenet',include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024,activation='relu')(x)
	x = Dense(1024,activation='relu')(x)
	out = Dense(5,activation='softmax')(x)

	model = Model(inputs=base_model.input,outputs=out)

	for layer in model.layers[:20]:
		layer.trainable=False
	for layer in model.layers[20:]:
		layer.trainable=True

	model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

	step_size_train = train_generator.n//train_generator.batch_size


	model.fit_generator(generator=train_generator,
						steps_per_epoch=step_size_train,
						epochs=5,
						verbose=1)
	return model


def crossValidation(nb_itr=1, validation_split=0.2):
	acc = 0

	for i in range(nb_itr):

		train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=validation_split)

		folder = "./data/"

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
																shuffle=True,
																subset='validation')

		model = train(train_generator)

		current_acc = model.evaluate(validation_generator,verbose=1)[1]
		print("Accuracy of iteration",i+1,":",round(current_acc*100,2),"%")
		acc += current_acc

	print("Average accuracy :",round((acc/nb_itr)*100,2),"%")


if __name__ == '__main__':
	crossValidation(5,0.2)
