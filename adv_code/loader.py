import os
import random
import torch
import numpy as np
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset


class ImageNet_Data(Dataset):
	def __init__(self, root_dir, label_name='.txt', folder_name='images'):
		labels_dir = os.path.join(root_dir, label_name)
		self.image_dir = os.path.join(root_dir, folder_name)
		self.labels = []

		with open(label_name, "r") as f:
			for line in f:
				line = line.strip("\n").rstrip()
				name_label = line.split(" ")
				self.labels.append(name_label[0], int(name_label[1]))

	def __len__(self):
		l = len(self.labels)
		return l

	def __getitem__(self, idx):
		filename = os.path.join(self.image_dir, self.labels[idx][0])
		in_img_t = cv2.imread(filename)[:, :, ::-1]
		
		in_img = np.transpose(in_img_t.astype(np.float32), axes=[2, 0, 1])
		img = in_img / 255.0

		label_true = self.labels[idx][1]
		# label_target = self.labels[idx][2]

		return img, label_true, filename
