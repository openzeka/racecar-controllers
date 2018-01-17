#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
import std_msgs.msg
from ackermann_msgs.msg import AckermannDriveStamped



rospy.init_node('opencv', anonymous=True)
pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
cmd = AckermannDriveStamped()

def get_object(hsv):
	hsv_lower_filter = (0, 147, 0)
	hsv_upper_filter = (22, 255, 255)
	mask = cv2.inRange(hsv, hsv_lower_filter, hsv_upper_filter)
	contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = 0
	Radius = 0
	if len(contours) > 0 :
		center_list = []
		radius_list = []
		for cnt in contours:
			(x,y),radius = cv2.minEnclosingCircle(cnt)
			center_list.append((int(x),int(y)))
			radius_list.append(int(radius))
		ix = np.argmax(radius_list)
		#print 'Nesne Konumu : ',center_list[ix]
		#print 'Nesne Buyuklugu : ',radius_list[ix]
		X = center_list[ix][0]	
		Radius = radius_list[ix]
		
	return	X,Radius



def talk():

    rate = rospy.Rate(20)

    cam_no = 1

    fps = 25
    delay = 1000/25

    vcap = cv2.VideoCapture(cam_no)
    

    while not rospy.is_shutdown():        
		ret,image = vcap.read()
		if ret == True:
			hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
			left = hsv[:,:672,:]
			right = hsv[:,672:,:]
		X = np.zeros(2)
		R = np.zeros(2)
		X[0],R[0] = get_object(left)
		X[1],R[1] = get_object(right)
		print 'Obejct : ' ,X.mean(),R.mean()
		angle = -2*(X.mean() - 336) / 336	
		if R.mean() >= 100 :
			speed = 0
		else:
			speed = 0.2
		print angle		
		cmd.drive.speed = speed					
		cmd.drive.steering_angle = angle
		pub.publish(cmd)
		rate.sleep()
        
        



if __name__ == '__main__':
    talk()
