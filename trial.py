
import matplotlib.pyplot as plt
import numpy as np


a = np.array([0, 1])
b = np.array([2, 1])


cos_theta = (a[0]*b[0] + a[1]*b[1]) / (np.linalg.norm(a) * np.linalg.norm(b))

print('cos_t:', cos_theta)

projection = np.linalg.norm(b) * cos_theta

print('projection:', projection)