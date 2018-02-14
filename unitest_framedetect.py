
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from fileOp.imgReader import ImageReader
from motion.motiondetector import MotionDetector
from annotation.pascal_voc import pacasl_voc_reader
from frameDetect.frameDetect import frameDetect

path = os.path.dirname(os.path.abspath(__file__))

bndboxWH = (224, 80)


def TestFrameDetect(bExtract=False):

	if bExtract==True:
		filePath = path+'/testcase/frameDetect/img72_0'
		voc = pacasl_voc_reader(filePath+'.xml')
		objectList = voc.getObjectList()
		img = cv2.imread(filePath+'.png')
		for (className, (xmin, ymin, xmax, ymax)) in objectList:
			roi = img[ymin:ymax+1, xmin:xmax+1]
			cv2.imwrite(path+'/testcase/frameDetect/'+className+'_03.png',roi)

	img0 = cv2.imread(path+'/testcase/frameDetect/'+'channel_00.png')
	img1 = cv2.imread(path+'/testcase/frameDetect/'+'channel_02.png')
	(rtn, (x, y, w, h)) = frameDetect(img0, img1, minW=200, minH=60, frameRatio=0.85)
	if rtn==True:
		print('frame detected {}'.format((x, y, w, h)))
		ch_focus_img = cv2.imread(path+'/testcase/frameDetect/'+'channel_00.png')
		cv2.rectangle(ch_focus_img, (x, y), (x+w, y+h), (255, 255, 255), 3)
		basics.showResizeImg(ch_focus_img, 'result', 0, width=0)
	else:
		print('frame not detected')

def main():
	testcase = TestFrameDetect(False)


main()