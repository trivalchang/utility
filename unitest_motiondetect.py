
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from fileOp.imgReader import ImageReader
from motion.motiondetector import MotionDetector

path = os.path.dirname(os.path.abspath(__file__))

class TestMotionDetect():

	framecnt = 0
	filePath = path+'/testcase/motionDetect/IMG_9083_small.m4v'

	imgReader = ImageReader(filePath, True)
	md = MotionDetector(0.4)

	(xmin, ymin, xmax, ymax) = (286, 98, 973, 561)
	while True: 
		(ret, frame, fname) = imgReader.read()
		if ret == False:
			break;
		framecnt = framecnt + 1

		#cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
		#key = basics.showResizeImg(frame, filePath, 1, width=800, height=600)

		roi = frame[ymin:ymax, xmin:xmax]
		key = basics.showResizeImg(roi, 'roi', 1, width=800, height=600)
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi = basics.blur_img(roi, 'gussian', (7,7))
		#key = basics.showResizeImg(roi, 'roi', 1, x=0, width=800)
		md.update(roi)
		if (framecnt > 16):
			result = md.detect(roi)
			if (result == None):
				continue
			(diff, cnts) = result
			key = basics.showResizeImg(diff, 'diff', 0, x=800, width=400)


		if key == ord('q'):
			break


def main():
	testcase = TestMotionDetect()


main()