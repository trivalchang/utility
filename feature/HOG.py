from __future__ import print_function

import numpy as np
import argparse
import glob
import cv2
import os 
import sys
import random
from skimage import feature

sys.path.insert(0, '/Users/developer/guru/')


class HOG():
	orientations = 9
	pixels_per_cell = (8, 8)
	cells_per_block = (2, 2)
	transform_sqrt = True 
	block_norm = 'L2'

	def __init__(self, _orientations, _pixels_per_cell, _cells_per_block, _transform_sqrt, _block_norm):
		self.orientations = _orientations
		self.pixels_per_cell = _pixels_per_cell
		self.cells_per_block = _cells_per_block
		self.transform_sqrt = _transform_sqrt
		self.block_norm = _block_norm

	def describe(self, img, bVisualize=False):
		if bVisualize == True:
			(hogWindow, hogImage) = feature.hog(img, 
											orientations=self.orientations, 
											pixels_per_cell=self.pixels_per_cell, 
											cells_per_block=self.cells_per_block, 
											transform_sqrt=self.transform_sqrt,
											block_norm=self.block_norm, 
											visualise=bVisualize)
			hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
			hogImage = hogImage.astype("uint8")	
			return (hogWindow, hogImage)
		else:
			return (feature.hog(img, 
							orientations=self.orientations, 
							pixels_per_cell=self.pixels_per_cell, 
							cells_per_block=self.cells_per_block, 
							transform_sqrt=self.transform_sqrt,
							block_norm=self.block_norm, 
							visualise=bVisualize), None)