import numpy as np
import operator
from sklearn.datasets.samples_generator import make_blobs

centers = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])
cur_center = np.mean(centers, 0)
print(cur_center)
