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
## -----------------------------------LOADING THE MODEL------------------------
model = load_model(PATH + 'src/latest-model-5.h5')
## -------------------------------------------UTILS----------------------------------------------
steering_img = cv2.imread(PATH + 'src/steering_wheel_image.png',0)
rows,cols = steering_img.shape

smoothed_angle = 0

#vid = cv2.VideoCapture(PATH + 'fgvt-data/GH010654.MP4')
#vid.set(cv2.CAP_PROP_POS_FRAMES, 25550)

## ---------------------------------------STARTING THE LOOP---------------------------------
i = 2000

pred_list = []
true_list = []
start = time.time()

try:
	while(cv2.waitKey(10) != ord('q')):
		orig_frame = cv2.imread(PATH+'autobahn/'+str(i)+'.jpg')
		#ret, orig_frame = vid.read()
		if orig_frame is not None:
			resized_frame = orig_frame[400:,:-400,:]
			resized_frame = cv2.GaussianBlur(resized_frame,(61,61),sigmaX=0,sigmaY=0)
			img = cv2.resize(resized_frame,(200,66))

			exp_img = np.expand_dims(img,0)
			pred = model.predict(exp_img)
			degrees = pred[0][0]
			print("Predicted angle: ", degrees)

			frame = cv2.resize(resized_frame,(640,480))  
			cv2.imshow('frame', frame)
			degrees = pred[0][0]
			smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
			M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
			dst = cv2.warpAffine(steering_img,M,(cols,rows))
			cv2.imshow('steering', dst)
		i += 1

except KeyboardInterrupt:
	cv2.destroyAllWindows()
