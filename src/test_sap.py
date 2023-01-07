import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import scipy
import time

## -------------------------------------IMPORTING DATASET-----------------

PATH = '/home/modi/sap/sap/'
datadir = PATH + 'autobahn/'

'''
columns = ['images']
data = pd.read_csv(datadir+'data.txt', names = columns)
data[['images','steering']] = data["images"].str.split(" ", 1, expand=True)
data['steering'] = np.array(data['steering'], dtype=float)
data['steering'] = data['steering'] * (scipy.pi/180)
'''
## -----------------------------------LOADING THE MODEL------------------------

model = load_model(PATH + 'src/fgvt-sap-model.h5')

## -------------------------------------------UTILS----------------------------------------------
steering_img = cv2.imread(PATH + 'src/steering_wheel_image.png',0)
rows,cols = steering_img.shape

smoothed_angle = 0

#vid = cv2.VideoCapture(PATH + 'fgvt-data/GH010654.MP4')
#vid.set(cv2.CAP_PROP_POS_FRAMES, 21060)

## ---------------------------------------STARTING THE LOOP---------------------------------
i = 1000

pred_list = []
true_list = []
start = time.time()

try:
	while(cv2.waitKey(10) != ord('q')):
	  orig_frame = cv2.imread(datadir+str(i)+'.jpg')
	  if orig_frame is not None:
		  img = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
		  img = cv2.blur(img, (51,51), cv2.BORDER_DEFAULT)
		  img = cv2.resize(img,(200,66))
		  img = img/255
		  
		  exp_img = np.expand_dims(img,0)
		  pred = model.predict(exp_img)
		  degrees = pred[0][0]
		  #pred_list.append(degrees)
		  #true_list.append(data['steering'][i])
		  print("Predicted steering angle: " + str(degrees) + " degrees")
		  #print("True steering angle: " + str(data['steering'][i]) + " degrees")
		  print('Time: ' + str(int(time.time()-start))+ 's')
		  

		  frame = cv2.resize(img,(360,240))  
		  cv2.imshow('frame', frame)
		  degrees = pred[0][0]*25
		  smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
		  M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
		  dst = cv2.warpAffine(steering_img,M,(cols,rows))
		  cv2.imshow('steering', dst)
	  i += 1

except KeyboardInterrupt:
	#plt.plot(pred_list, 'r', label='predicted angle')
	#plt.plot(true_list, 'b', label='true angle')
	#plt.legend()
	#plt.show()
	cv2.destroyAllWindows()
