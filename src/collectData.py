import pygame
import time
import psutil
import cv2

PATH = '/home/modi/sap/sap/'


pygame.init()
pygame.joystick.init()
joysticks = {}

		
def get_angle():
	for event in pygame.event.get():
		if event.type == pygame.JOYDEVICEADDED:
			joy = pygame.joystick.Joystick(event.device_index)
			joysticks[joy.get_instance_id()] = joy	
	for joystick in joysticks.values():
		return joystick.get_axis(3)
	    	
def main():
	try:
		i = 700
		smoothed_angle = 1
		while True:
			orig_frame = cv2.imread(PATH+'autobahn/'+str(i)+'.jpg')
			orig_frame = cv2.resize(orig_frame,(640,480))
			steering_img = cv2.imread(PATH + 'src/steering_wheel_image.png',0)
			rows,cols = steering_img.shape
			degrees = get_angle()*2
			print(degrees)
			smoothed_angle += 0.2 * pow(abs((degrees*25 - smoothed_angle)), 2.0 / 3.0) * (degrees*25 - smoothed_angle) / abs(degrees*25 - smoothed_angle)
			M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
			dst = cv2.warpAffine(steering_img,M,(cols,rows))
			
			cv2.imshow('road', orig_frame)
			cv2.imshow('steering', dst)
			cv2.waitKey(10)
			i += 1
			#time.sleep(0.1)
		
	except KeyboardInterrupt:
    		PROCNAME = "python3"
    		for proc in psutil.process_iter():
    			if proc.name() == PROCNAME:
    				proc.kill()

if __name__=='__main__':
	main()
	
