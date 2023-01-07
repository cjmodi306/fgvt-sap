#https://github.com/rslim087a/Self-Driving-Car-Course-Codes/blob/master/Section%205%20Resources%20(Finding%20Lanes)/Source%20Code/lanes.py
#https://www.youtube.com/watch?v=eLTLtUVuuy4&list=RDCMUCs6nmQViDpUw0nuIx9c_WvA&start_radio=1&rv=eLTLtUVuuy4&t=5041

import cv2
import numpy as np
import sys
import logging

PATH = '/home/modi/sap/sap/'
datadir = PATH + 'autobahn/'

def detect_edges(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(15,15),0,0)
	edges = cv2.Canny(blur, 50, 100)
	#cv2.imshow("lane lines", cv2.resize(edges,(640,480)))
	#cv2.waitKey(3000)
	return edges
	
def region_of_interest(edges, frame):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # only focus bottom half of the screen
    polygon = np.array([
        [(0, height-250),(0, height),(width, height),(900, int(height/2)-100),(700, int(height/2)-100)]
    ])
    
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges,mask)
    #cv2.imshow("lane lines", cv2.resize(cropped_edges,(640,480)))
    #cv2.waitKey(2000)
    return cropped_edges

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=4, maxLineGap=400)
    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

if __name__ == '__main__':
	i = 2500
	while True:
		frame = cv2.imread(datadir+str(i)+'.jpg')
		frame = frame[500:-100,:,:]
		edges = detect_edges(frame)
		cropped_edges = region_of_interest(edges, frame)
		line_segments = detect_line_segments(cropped_edges)
		lane_lines_image = display_lines(frame, line_segments)
		lane_lines_image = cv2.resize(lane_lines_image, (640,480))
		cv2.imshow("lane lines", lane_lines_image)
		cv2.waitKey(10)
		i+=1
