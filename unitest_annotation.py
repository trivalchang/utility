
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from annotation.pascal_voc import pacasl_voc_reader, pacasl_voc_writer

path = os.path.dirname(os.path.abspath(__file__))

class TestPascalVoc():
	imgPath = path+'/testcase/annotation_pascal/img77.jpg'

	img = cv2.imread(imgPath)
	voc_writer = pacasl_voc_writer(imgPath.replace('.jpeg', '').replace('.jpg', ''), 
									'test', img.shape)
	voc_writer.new_box('test1', (100, 300, 300, 400))
	voc_writer.new_box('test2', (500, 600, 700, 800))
	voc_writer.save()

	xmlFname = imgPath.replace('.jpeg', '').replace('.jpg', '')+'.xml'
	voc_reader = pacasl_voc_reader(xmlFname)
	(fname, width, height, depth) = voc_reader.imageInfo()
	print('fname = {}, {}x{}x{}'.format(fname, width, height, depth))
	objList = voc_reader.getObjectList()
	for (name, (xmin, ymin, xmax, ymax)) in objList:
		print('{} : {} {} {} {}'.format(name, xmin, xmax, ymin, ymax))

def main():
	testcase = TestPascalVoc()


main()