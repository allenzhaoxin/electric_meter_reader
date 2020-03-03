#!/usr/bin/env python
# coding: utf-8

# In[481]:

import cv2
import numpy as np
import os

class Crop:
	def __init__(self):
		self.name = "Crop"
	
	@staticmethod
	def pre_process_binary(img_):
		img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
		img_ = cv2.bilateralFilter(img_,9,75,75)
		img_ = cv2.equalizeHist(img_)
		img_ = cv2.bitwise_not(img_)
		thresh, img_binary = cv2.threshold(img_, 150, maxval=255, type=cv2.THRESH_BINARY)
		kernel = np.ones((1,3), np.uint8) 
		img_binary = cv2.dilate(img_binary, kernel, iterations=3) 
		return img_binary
		
	@staticmethod
	def pre_process_hsv(img):
		#plt.imshow(img)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		lower_red_1 = np.array([0, 50, 70]) 
		upper_red_1 = np.array([7, 255, 255])
		lower_red_2 = np.array([170, 50, 70]) 
		upper_red_2 = np.array([179, 255, 255])
		mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
		mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
		mask = mask_1 | mask_2
		return mask
	
	def remove_noise(self, img_):
		ret, labels = cv2.connectedComponents(img_)
		count = [0] * ret
		max_i = [0] * ret
		min_i = [10000] * ret
		#start_time = time.time()
		#print(labels.shape)

		for i in range(0, labels.shape[0]):
			for j in range(0, labels.shape[1]):
				lb = labels[i,j]
				if (max_i[lb] < j):
					max_i[lb] = j
				if (min_i[lb] > j):
					min_i[lb] = j
				count[lb] = count[lb] + 1 
		count[0] = 0
		max_value = 0
		max_index = 0
		for i in range(0, ret):
			if (count[i] >= max_value):
				max_value = count[i]
				max_index = i
				
		labels[labels != max_index] = 0
		labels[labels != 0] = 1
		#print(labels.shape)
		#print("Total run_time = ", time.time() - start_time)
		return max_i[max_index], min_i[max_index]
	
	@staticmethod
	def main_process(self, img, img_binary, mask):
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
		sizes = stats[1:, -1]; nb_components = nb_components - 1
		min_size = output.shape[0]*output.shape[1]//3500
		max_size = output.shape[0]*output.shape[1]//500
		#your answer image
		x_min = 10000
		red_point = [100, 100]
		open_range = output.shape[0]//13
		modify_range = 20
		#for every component in the image, you keep it only if it's above min_size
		for i in range(0, nb_components - 1):
			if sizes[i] >= min_size and sizes[i] <= max_size:
				rec_min = int(centroids[i + 1][1]) - open_range
				rec_max = int(centroids[i + 1][1]) + open_range
				if (np.count_nonzero(img_binary[rec_min:rec_max, :])/(open_range*2*output.shape[1]) >= 0.3 and centroids[i + 1][1] < x_min):
					red_point = centroids[i + 1]
					x_min = centroids[i + 1][1]

		img3 = img.copy()
		conner_left = (max(0,int(red_point[0]) - output.shape[1]*10//20), max(0,int(red_point[1]) - open_range))
		conner_right = (min(int(red_point[0]) + output.shape[1]//30,img3.shape[1]), min(int(red_point[1]) + open_range, img3.shape[0]))
		
		max_i, min_i = self.remove_noise(img_binary[conner_left[1]:conner_right[1], conner_left[0]:conner_right[0]])

		conner_left = (conner_left[0] + min_i, conner_left[1])
		#conner_right = (conner_left[0] + max_i, conner_right[1])

		conner_left_modify = (conner_left[0], int(red_point[1]) - modify_range)
		conner_right_modify = (conner_right[0], int(red_point[1]) + modify_range*2)
		
		img4 = img_binary[conner_left_modify[1]:conner_right_modify[1], conner_left_modify[0]+10:conner_right_modify[0]]
		kernel = np.ones((5,5), np.uint8) 
		img4[:,:50] = cv2.dilate(img4[:,:50], kernel, iterations=5) 
		#img4 = self.remove_noise(img4)
		save_modify = 0
		for i in range(50, img4.shape[1], 50):
			#print(conner_right[1] - conner_left[1])
			if (conner_right[1] - conner_left[1])/(conner_right[0] - conner_left[0] - i) > 0.6:
				save_modify = i - 50
				break
			if (np.count_nonzero(img4[:,i-50:i]) < 50*modify_range*3 - 50):
				save_modify = i - 50
				break
		#print(img4.shape)
		#save_modify = 0
		img5 = img3[conner_left[1]:conner_right[1], conner_left[0] + save_modify :conner_right[0]]
		return img5
		#plt.imshow(img5)
	
	def run(self, img):
		#plt.imshow(img)
		img_binary = self.pre_process_binary(img)
		mask = self.pre_process_hsv(img)
		final_img = self.main_process(self, img, img_binary, mask)
		return final_img




