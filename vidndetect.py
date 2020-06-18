

import cv2 as cv
import numpy as np
import detectcircle as circ
import tensorflow as tf
import os
from playsound import playsound

print("Tensorflow:", tf.__version__)
print("OpenCV:", cv.__version__)
CLASS = ['30km/h','40km/h','Not Speed Sign']
aud_path = './Audio/'
file_path = './new/'
files = os.listdir(file_path)
count = len(files)
n = 0
totalframe = 0
cap =cv.VideoCapture('siang1.mp4')
hasil_class = ''

def load_img(path):
    #print(path)
    return cv.imread(path)

def preprocess_image(img, side=50):
    min_side = min(img.shape[0], img.shape[1])
    img = img[:min_side, :min_side]
    img = cv.resize(img, (side,side))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img / 255.0

while(True):
    ret, frame = cap.read()
    if(frame is None):
        print('video frame:' + str(totalframe))
        break
    totalframe += 1 
    h , w , layers =  frame.shape
    new_h = int(h/2)
    new_w = int(w/2)
    M = cv.getRotationMatrix2D((w/2,h/2),270,1)
    dst = cv.warpAffine(frame,M,(w,h))
    img = dst.copy()*0
    dst1 = dst.copy()
    circle = circ.process_img(dst)
    
    wi = 0
    ha = 0
    x = 0 
    y = 0
      
    if(circle is not None): 
        count+=1
        for i in circle[0,:]:
            #draw the outer circle
            cv.circle(dst,(i[0],i[1]),i[2],(0,255,0),2)
            wi = i[2]*2
            ha = i[2]*2
            x = i[1]-i[2]
            y = i[0]-i[2]
        img = cv.resize(img, (wi,ha))
        for i in range(wi):
            for j in range(ha):
                img[i,j] = dst1[x+i,y+j]
        img = cv.resize(img, (50,50))
        cv.imwrite('./Test/data_'+str(count)+'.png',img)
        file_im = './Test/data_'+str(count)+'.png'
        dst = cv.resize(dst, (1280,720))
        cv.imshow('frame', dst)
        if(totalframe == 23): 
            cv.imwrite('./Test/fr_'+str(count)+'.png',dst)

        test_images = [load_img(file_path + file) for file in files]
        for i in range(len(test_images)):
            test_images[i] = preprocess_image(test_images[i])
        test_images = np.expand_dims(test_images, axis=-1)
        layers = [
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=test_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Dropout(0.5),               
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=5, activation=tf.nn.softmax)
        ]
        model = tf.keras.Sequential(layers)
        model.load_weights("ww1.tf")

        eval_images = [preprocess_image(load_img(file_im))]
        eval_images = np.expand_dims(eval_images, axis=-1)
        predictions = model.predict([eval_images])
        #rekognisi = np.argmax(predictions)
        
        for i in range(len(eval_images)):
            tmp_class = CLASS[np.argmax(predictions[i])] 
            print("\n\n")
            print(tmp_class)
            print("\n\n")
            if (tmp_class != 'Not Speed Sign' and hasil_class != tmp_class):
                hasil_class = tmp_class
                class_id = np.argmax(predictions[i]) 
                percen = np.max(predictions[i])
                if (class_id==0 and percen>0.4): playsound(aud_path + '30.mp3')
                elif (class_id==1 and percen>0.4): playsound(aud_path + '40.mp3')
    else:
        dst = cv.resize(dst, (1280,720))
        cv.imshow('frame', dst)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        print('video frame:' + str(totalframe))
        break
    
cap.release()
cv.destroyAllWindows()