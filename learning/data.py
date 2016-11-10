import os
from PIL import Image
import numpy as np

def loadData():
	data = np.empty((42000, 1, 28, 28), dtype = "float32")
	label = np.empty((42000, ), dtype = "uint8")

	files = os.listdir("./mnist")
	num = len(files)
	for i in range(num):
		image = Image.open("./mnist/" + files[i])
		data[i, :, :, :] = np.asarray(image, dtype = "float32")
		label[i] = int(files[i].split('.')[0])
	return data, label
