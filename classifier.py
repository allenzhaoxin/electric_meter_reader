#!/usr/bin/env python
# coding: utf-8

# In[481]:


import os, sys
import cv2
import numpy as np
import math
import pandas as pd

class Classifier:
	def __init__(self, model):
		self.name = "Classifier"
		self.model = model
	
	@staticmethod
	def rotate_bound(image, angle):
		# grab the dimensions of the image and then determine the
		# center
		(h, w) = image.shape[:2]
		(cX, cY) = (w / 2, h / 2)

		# grab the rotation matrix (applying the negative of the
		# angle to rotate clockwise), then grab the sine and cosine
		# (i.e., the rotation components of the matrix)
		M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])

		# compute the new bounding dimensions of the image
		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))

		# adjust the rotation matrix to take into account translation
		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY

		# perform the actual rotation and return the image
		return cv2.warpAffine(image, M, (nW, nH))
	
	@staticmethod
	def pre_process(self, img):
		#print(img)
		img = cv2.equalizeHist(img)
		img = cv2.medianBlur(img, 9)
		
		edges = cv2.Canny(img, 50, 150)
		#rotate image
		threshold = 50
		check = False
		#plt.imshow(img)
		while (not check or lines is None or lines.shape[0] > 10):
			check = True
			lines = cv2.HoughLines(edges,1,np.pi/180,threshold)
			#print(lines)
			threshold = threshold + 10
		#print(lines.shape)
		img_line = img.copy()
		rotate_angle = -1
		
		trust = [0]*lines.shape[0]
		for i in range (0, lines.shape[0]):
			angle1 = 90-lines[i][0][1]*180/np.pi
			for j in range (i+1, lines.shape[0]):
				angle2 = 90-lines[j][0][1]*180/np.pi
				if (abs(angle1 - angle2) <= 2):
					trust[i] = trust[i] + 1
		index = 0

		for line in lines:
			if (trust[index] < 3):
				continue
			else:
				rho, theta = line[0]
				#print(theta)
				if (theta < np.pi*2/3 and theta > np.pi/4):
					rotate_angle = max(rotate_angle, theta)
			index = index + 1
		rotated = img_line
		const_cut = 0
		if (rotate_angle != -1):
			rotated = self.rotate_bound(img_line, 90-rotate_angle*180/np.pi)
			const_cut = abs(90-rotate_angle*180/np.pi)/3
		rotated = rotated[:, :rotated.shape[1]-rotated.shape[1]//6-10]

		#remove redundant part
		img_cut = rotated.copy()
		y_ed = img_cut.shape[0]
		y_st = 0
		img_cut = img_cut[y_st+int(img_cut.shape[0]*const_cut/8):y_ed-int(img_cut.shape[0]*const_cut/8),:]
		#img_cut = img_cut[:,70:img_cut.shape[1] - img_cut.shape[1]//10]
		#plt.imshow(img_cut)
		return img_cut
	
	@staticmethod
	def convert_to_binary(img_grayscale, thresh=100):
		thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
		return img_binary
	
	@staticmethod
	def remove_noise(img_):
		ret, labels = cv2.connectedComponents(img_)
		if (ret <= 1):
			return img_
		count = [0] * ret
		max_i = [0] * ret
		min_i = [1000] * ret
		for i in range(0, labels.shape[0]):
			for j in range(0, labels.shape[1]):
				lb = labels[i,j]
				if (max_i[lb] < i):
					max_i[lb] = i
				if (min_i[lb] > i):
					min_i[lb] = i
				count[lb] = count[lb] + 1 
		count[0] = 0
		max_value = 0
		max_index = 0
		for i in range(0, ret):
			if (count[i]*(max_i[i]-min_i[i]) >= max_value):
				max_value = count[i]*(max_i[i]-min_i[i])
				max_index = i
		labels[labels != max_index] = 0
		labels[labels != 0] = 1
		#print(labels.shape)
		return img_*labels
	
	@staticmethod
	def remove_remain_right_edge(img_):
		ret, labels = cv2.connectedComponents(img_)
		if (ret <= 1):
			return img_
		count = 0
		edge_label = 0
		for i in range(0, labels.shape[0]):
			if (labels[i,labels.shape[1] - 1] != 0):
				edge_label = labels[i,labels.shape[1] - 1]
				count = count + 1
		#print(labels.shape[1])
		if (count / labels.shape[0] > 0.65):
			for i in range(0, labels.shape[0]):
				for j in range(0, labels.shape[1]):
					if (labels[i,j] == edge_label and j >= labels.shape[1]*19//20):
						labels[i,j] = 0
		labels[labels != 0] = 1
		#print(labels)
		return img_*labels
	
	@staticmethod
	def remove_remain_bottom_edge(img_):
		ret, labels = cv2.connectedComponents(img_)
		if (ret <= 1):
			return img_
		count_bot = 0
		count_top = 0
		edge_label_bot = 0
		edge_label_top = 0
		for j in range(0, labels.shape[1]):
			if (labels[labels.shape[0] - 1, j] != 0):
				edge_label_bot = labels[labels.shape[0] - 1, j]
				count_bot = count_bot + 1
			if (labels[0, j] != 0):
				edge_label_top = labels[0, j]
				count_top = count_top + 1
		#print(count_top)
		if (count_bot / labels.shape[1] > 0.4):
			for i in range(0, labels.shape[0]):
				for j in range(0, labels.shape[1]):
					if (labels[i,j] == edge_label_bot):
						labels[i,j] = 0
		if (count_top / labels.shape[1] > 0.4):
			for i in range(0, labels.shape[0]):
				for j in range(0, labels.shape[1]):
					if (labels[i,j] == edge_label_top):
						labels[i,j] = 0
		labels[labels != 0] = 1
		return img_*labels
	
	@staticmethod
	def split_img(self, img_binary):
		mini_img = []
		kernel = np.ones((2,2), np.uint8)
		rm_thresh = 5
		for i in range(0,5):
			st = i*img_binary.shape[1]//5
			ed = (i+1)*img_binary.shape[1]//5
			mini_img.append(img_binary[:, st:ed])
			mini_img[i] = self.remove_noise(mini_img[i])
			mini_img[i] = np.array(mini_img[i], dtype='uint8')
		return mini_img
	
	@staticmethod
	def normalize_img(mini_img):
		rm_thresh = 5
		st_cut_0 = 0
		ed_cut_0 = mini_img.shape[0]-1
		st_cut_1 = 0
		ed_cut_1 = mini_img.shape[1]-1
		for j in range(0, mini_img.shape[0]):
			if (np.count_nonzero(mini_img[j]) > rm_thresh):
				st_cut_0 = j
				break
		for j in range(mini_img.shape[0]-1, 0, -1):
			if (np.count_nonzero(mini_img[j]) > rm_thresh):
				ed_cut_0 = j
				break
		ed_cut_0 = max(ed_cut_0, st_cut_0 + 1)
		mini_img = mini_img[st_cut_0:ed_cut_0, :]

		for j in range(0, mini_img.shape[1]):
			if (np.count_nonzero(mini_img[:,j]) > rm_thresh):
				st_cut_1 = j
				break
		for j in range(mini_img.shape[1]-1, 0, -1):
			if (np.count_nonzero(mini_img[:,j]) > rm_thresh):
				ed_cut_1 = j
				break
		ed_cut_1 = max(ed_cut_1, st_cut_1 + 1)
		mini_img = mini_img[:,st_cut_1:ed_cut_1]
		mini_img = cv2.copyMakeBorder(mini_img, 0, 0, mini_img.shape[1]*1//3, mini_img.shape[1]*1//3, cv2.BORDER_CONSTANT)
		mini_img = cv2.resize(mini_img, dsize=(28, 28))
		mini_img = mini_img.astype('float32') / 255
		return mini_img
	
	@staticmethod
	def main_process(self, img_cut, model):
		min_threshold = 130
		max_threshold = 180
		step = 5
		result = 0
		predict_num = [ [0] * 10 for _ in range(5)]
		predict_num[0][0] = 10
		predict_num[1][0] = 5
		for threshold in range(min_threshold, max_threshold, step):
			img_binary = self.convert_to_binary(img_cut, threshold)
			#plt.imshow(img_binary)
			img_binary = self.remove_remain_right_edge(img_binary)
			img_binary = np.array(img_binary, dtype='uint8')
			img_binary = self.remove_remain_bottom_edge(img_binary)
			img_binary = np.array(img_binary, dtype='uint8')
			coords = cv2.findNonZero(img_binary) # Find all non-zero points (text)
			x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
			img_binary = img_binary[y:y+h, x:x+w]
			if (img_binary.shape[0] == 0 or img_binary.shape[1] == 0):
				continue
			mini_img = self.split_img(self, img_binary)
			
			for i in range(0,5):
				ratio = np.count_nonzero(mini_img[i])/(mini_img[i].shape[0]*mini_img[i].shape[1])
				if (ratio <= 0.3 and ratio >= 0.05):
					mini_img[i] = self.normalize_img(mini_img[i])
					pred = model.predict([mini_img[i].ravel()])
					predict_num[i][pred[0]] = predict_num[i][pred[0]] + 1
		#print(predict_num)
		for i in range(0,5):
			result = result*10 + np.array(predict_num[i], dtype='uint8').argmax()
		return result
	
	def run(self, img):
		img_cut = self.pre_process(self, img)
		final_res = self.main_process(self, img_cut, self.model)
		return final_res




