
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from fileOp.imgReader import ImageReader
from motion.motiondetector import MotionDetector
from annotation.pascal_voc import pacasl_voc_reader
from frameDetect.frameDetect import FrameDetectByOneImage, FrameDetectByDiffImages

path = os.path.dirname(os.path.abspath(__file__))

bndboxWH = (224, 80)

def ExtractClassImg(filePath, outputPostFix):
	voc = pacasl_voc_reader(filePath+'.xml')
	objectList = voc.getObjectList()
	img = cv2.imread(filePath+'.png')
	for (className, (xmin, ymin, xmax, ymax)) in objectList:
		roi = img[ymin:ymax+1, xmin:xmax+1]
		cv2.imwrite(path+'/testcase/frameDetect/'+className+outputPostFix+'.png',roi)	

def TestFrameDetect(bExtract=False):

	img0 = cv2.imread(path+'/testcase/frameDetect/'+'channel_00.png')
	img0Origin = img0.copy()
	img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	(rtn, (x, y, w, h)) = FrameDetectByOneImage(img0Origin, img0, minW=200, minH=60, frameRatio=0.85)
	if rtn==True:
		print('frame detected {}'.format((x, y, w, h)))
		ch_focus_img = cv2.imread(path+'/testcase/frameDetect/'+'channel_00.png')
		cv2.rectangle(ch_focus_img, (x, y), (x+w, y+h), (255, 255, 255), 3)
		basics.showResizeImg(ch_focus_img, 'result', 0, width=0)
	else:
		print('frame not detected')

def TestFrameDetectByDiffImages():

	img0 = cv2.imread(path+'/testcase/frameDetect/'+'mainMenu_channel_focus.png')
	img1 = cv2.imread(path+'/testcase/frameDetect/'+'mainMenu_channel_unfocus.png')
	img0Origin = img0.copy()
	img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	img0 = basics.threshold_img(img0, 'OTSU', False)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img1 = basics.threshold_img(img1, 'OTSU', False)

	(rtn, (x, y, w, h)) = FrameDetectByDiffImages(img0Origin, img0, img1, minW=200, minH=60, frameRatio=0.85)
	if rtn==True:
		print('diff frame detected {}'.format((x, y, w, h)))
		ch_focus_img = cv2.imread(path+'/testcase/frameDetect/'+'mainMenu_channel_focus.png')
		cv2.rectangle(ch_focus_img, (x, y), (x+w, y+h), (255, 255, 255), 3)
		basics.showResizeImg(ch_focus_img, 'result', 0, width=0)
	else:
		print('frame not detected')

def main():
	if (False):
		ExtractClassImg(path+'/testcase/frameDetect/'+'img132_0', '_channel_unfocus')
		ExtractClassImg(path+'/testcase/frameDetect/'+'img44_0', '_channel_focus')
	#testcase = TestFrameDetect(False)
	testcase = TestFrameDetectByDiffImages()


main()