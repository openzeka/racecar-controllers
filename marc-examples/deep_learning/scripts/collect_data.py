#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2				# opencv kütüphanesi
import numpy as np
import os 
import time

import rospkg
import sys
import rospy
from ackermann_msgs.msg import AckermannDriveStamped	# motor mesajı

# ackermann 
ackermann_msg = AckermannDriveStamped()
def drive_callback(msg):
	global ackermann_msg
	ackermann_msg = msg


rospack = rospkg.RosPack()
# rospack.list_pkgs() 
package_path = rospack.get_path('deep_learning')
if not os.path.exists(package_path + '/data/'):
    os.makedirs(package_path + '/data/')

class seyir_logger(object):
	def __init__(self,cam_no=1):
		self.vcap = cv2.VideoCapture(cam_no)
		path = package_path + '/data/'
		i = 1
		while True:
			dname = path+'%03d'%i
			if os.path.exists(dname):
				i += 1 
			else:
				os.makedirs(dname)
				break
		self.path = dname+'/'
		self.filep = file(self.path+'seyir.csv', "w+") 
		self.filep.write('FileName,Speed,Angle\n')
		self.index = 0		
	
	def write(self,speed,angle):
		ret, frame = self.vcap.read()
		if ret :
			fname = self.path+'%05d.jpg'%self.index
			cv2.imwrite(fname,frame)
			line = fname +','+ str(speed)+ ','+str(angle)+'\n'
			self.filep.write(line)
			self.index += 1
		else:
			print "Frame not read !... "
			
	def close(self):
		self.vcap.release()
		self.filep.close()
	

if __name__ == '__main__':
	global ackermann_msg
	logger = seyir_logger()
	rospy.init_node('collect_data')
	ackermann_sub = rospy.Subscriber('/ackermann_cmd',AckermannDriveStamped, drive_callback, queue_size=1)
	while not rospy.is_shutdown():
		time.sleep(0.04)
		if not ackermann_msg.drive.speed == 0:
			logger.write(ackermann_msg.drive.speed, ackermann_msg.drive.steering_angle)
		print(ackermann_msg.drive.speed, "  ", ackermann_msg.drive.steering_angle)
	rospy.spin()
			
	logger.close()
