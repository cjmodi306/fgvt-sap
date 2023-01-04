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
vid = cv2.VideoCapture(PATH + 'fgvt-data/GH010655.MP4')
#vid.set(cv2.CAP_PROP_POS_FRAMES, 23550)

## ---------------------------------------STARTING THE LOOP---------------------------------
i = 0
count = 0
pred_list = []
true_list = []
start = time.time()

try:
	while(cv2.waitKey(10) != ord('q')):
		ret, orig_frame = vid.read()
		if orig_frame is not None and count%10==0:
			cv2.imwrite(PATH+'autobahn/'+str(i+3886)+'.jpg', orig_frame)
			i += 1
		else:
			print(orig_frame)
		count += 1
		print(i)
except KeyboardInterrupt:
	cv2.destroyAllWindows()
