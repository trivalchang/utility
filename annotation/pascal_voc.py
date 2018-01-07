# import the necessary packages

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs


class pacasl_voc_reader:
	root = None
	def __init__(self, fname):
		tree = ElementTree.parse(fname)
		self.root = tree.getroot()

	def imageInfo(self):

		try:
			fname = self.root.find('filename').text
			size = self.root.find('size')
			width = int(size.find('width').text)
			height = int(size.find('height').text)
			depth = int(size.find('depth').text)
		except:
			fname = None
			width = 0
			height = 0
			depth = 0
		return(fname, width, height, depth)


	def getObjectList(self):
		objectList = []
		for obj in self.root.findall('object'):
			try:
				name = obj.find('name').text
				bndbox = obj.find('bndbox')
				xmin = int(bndbox.find('xmin').text)
				xmax = int(bndbox.find('xmax').text)
				ymin = int(bndbox.find('ymin').text)
				ymax = int(bndbox.find('ymax').text)
				objectList.append((name, (xmin, ymin, xmax, ymax)))
			except:
				continue
		return objectList


class pacasl_voc_writer:

	def __init__(self, fName, folder, imgSize, path=None):
		self.fName = fName
		self.imgSize = imgSize
		self.folder = folder
		self.path = path
		self.objList = []

	def new_box(self, objName, box):
		self.objList.append((objName, box))

	def prettify(self, elem):
		rough_string = ElementTree.tostring(elem, 'utf8')
		root = etree.fromstring(rough_string)
		return etree.tostring(root, pretty_print=True)

	def save(self):
		vocFile = codecs.open(self.fName+'.xml', 'w')
		annotation = Element('annotation')
		annotation.set('verified', 'yes')
		folder = SubElement(annotation, 'folder')
		folder.text = self.folder
		filename = SubElement(annotation, 'filename')
		filename.text = self.fName
		if (self.path != None):
			path = SubElement(annotation, 'path')
			path.text = self.path

		source = SubElement(annotation, 'source')
		database = SubElement(source, 'database')
		database.text = 'Unkown'

		size = SubElement(annotation, 'size')
		width = SubElement(size, 'width')
		width.text = str(self.imgSize[1])
		height = SubElement(size, 'height')
		height.text = str(self.imgSize[0])
		depth = SubElement(size, 'depth')
		depth.text = str(self.imgSize[2])

		for (objName, objBox) in self.objList:
			print(objName, objBox)
			obj = SubElement(annotation, 'object')
			name = SubElement(obj, 'name')
			name.text = objName
			pose = SubElement(obj, 'pose')
			pose.text = 'Unspecified'
			truncated = SubElement(obj, 'truncated')
			truncated.text = '0'
			difficult = SubElement(obj, 'difficult')
			difficult.text = '0'
			bndbox = SubElement(obj, 'bndbox')
			xmin = SubElement(bndbox, 'xmin')
			xmin.text = str(objBox[0])
			ymin = SubElement(bndbox, 'ymin')
			ymin.text = str(objBox[1])
			xmax = SubElement(bndbox, 'xmax')
			xmax.text = str(objBox[2])
			ymax = SubElement(bndbox, 'ymax')
			ymax.text = str(objBox[3])

		prettifyResult = self.prettify(annotation)
		vocFile.write(prettifyResult.decode('utf8'))
		vocFile.close()