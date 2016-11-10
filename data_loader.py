import os
import numpy as np

def load_data():
	file = 'data.txt'
	input = open(file, 'r')
	for i in range(860):
		line = input.readline()
	data = np.empty((201,2,37,65), dtype = "int32")
	out = np.empty((201,63,), dtype = "float32")
	for i in range(201):
		line = input.readline()
		coors = line.split(' ')
		for j in range(63):
			out[i,j] = coors[37*65*2+1+j]
		for j in range(2):
			for k in range(37):
				for m in range(65):
					v = int(coors[k*65*2+m*2+j+1])
					if v == -1:
						v = 0
					data[i,j,k,m] = v

	return data,out;