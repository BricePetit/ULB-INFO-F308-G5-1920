"""
les cables pour les moteurs, le cable brun est vers l'exterieur

raspivid -o - -t 0 -hf -w 800 -h 400 -fps 24 |cvlc -vvv stream:///dev/stdin --sout '#standard{access=http,mux=ts,dst=:8160}' :demux=h264
"""
# Import pour le sleep
import time

# Import the PCA9685 module.
import Adafruit_PCA9685

# Import lib pour camera
import picamera

# Import pour prendre en charge l'input du bouton
import RPi.GPIO as GPIO

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing import image
import numpy as np

def openCloseTrash(number_trash):
    """
    Ouverture d'une poubelle avec un certain numero et ferme la poubelle apres 5 sec
    """
    # Configure min and max servo pulse lengths
    servo_min = 120  # Min pulse length out of 4096
    servo_max = 600  # Max pulse length out of 4096
    pwm.set_pwm(number_trash, 0, servo_min)
    time.sleep(5)
    pwm.set_pwm(number_trash, 0, servo_max)

def takePic():
    """
    Allume la camera, prends un photo et eteint la camera
    """
    camera = picamera.PiCamera() #creation de l'instance camera
    try:
        camera.resolution = (1920, 1080)
        #camera.framerate = 32
        #camera.vflip = True # faire une rotation verticale
        camera.capture('dechet.jpg', resize=(424,552)) #prendre photo
    finally:
        camera.close()

def predict(img_path):
    categories = ["Blanc","Bleu","Jaune","Orange","Verre"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    result = model.predict([img])
    return categories[np.argmax(result[0])]

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)
# Regarder si ca change la vitesse d'Ouverture

# Config boutton
GPIO.setmode(GPIO.BCM)
button_input = 17 #11 pour GPIO.BOARD car c'est le num 11 phyisquement et 17 logiquement
GPIO.setup(button_input, GPIO.IN, pull_up_down=GPIO.PUD_UP)

model = load_model('Model.model')

try:
   while True:
        # lorsque le boutton est presse
        if (not GPIO.input(button_input)):
            time.sleep(0.5)
            #takePic()
            res = predict("dechet.jpg")
            categories = ["Blanc","Bleu","Jaune","Orange","Verre"]
            index = categories.index(res)
            openCloseTrash(index)
            GPIO.cleanup()
except KeyboardInterrupt:
    GPIO.cleanup()
