#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, struct, array, time
from fcntl import ioctl

from sensor_msgs.msg import Joy

import cv2
import numpy as np
import rospy
import threading
from enum import Enum
import std_msgs.msg

import os.path
from os.path import expanduser
import re
from ackermann_msgs.msg import AckermannDriveStamped
# Keras and OpenCv
import cv2
import json
from keras.models import model_from_json
	
# mname.json for model structure 
# mname.h5 for model weights
	
model_name = '/home/nvidia/racecar-ws/src/racecar-controllers/deep_learning/scripts/model_new'
jstr = json.loads(open(model_name+'.json').read())
model = model_from_json(jstr)
model.load_weights(model_name+'.h5')
#

rospy.init_node('predict', anonymous=True)
pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)


def predict(image):
    steering = 0.0
    speed = 0.7
    out = model.predict(image, batch_size=1)
    steering = out[0][0]
	
    return {'steering':steering, 'speed': speed}



def talk():

    rate = rospy.Rate(20)

    cam_no = 1
    fps = 25
    delay = 1000/25
    vcap = cv2.VideoCapture(cam_no)

    while True:
	
		ret,image = vcap.read()
		if ret == True:
			prediction = predict(image.reshape(1,376,1344,3))
		msg = AckermannDriveStamped()
		
		msg.drive.speed = prediction['speed']
		msg.drive.steering_angle = prediction['steering']
		
		pub.publish(msg)

		rate.sleep()

    vcap.release()
	
if __name__ == '__main__':
    talk()
