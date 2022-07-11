import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import os

os.chdir('C:/Users/itwill/Desktop/apple_1')
path = "./"
file_lst = os.listdir(path)
print(file_lst)

size_lst = []
for file in file_lst:
    size_lst.append(plt.imread(file).shape)

from collections import Counter
cnt = Counter(size_lst)
print(cnt) # Counter({(3096, 4128, 3): 37, (2592, 3888, 3): 6, (3648, 5472, 3): 4, (3024, 4032, 3): 1})

apple_lst = []
for file in file_lst:
    apple_lst.append(plt.imread(file))

import cv2
#test = apple_lst[0]
#test_rs = cv2.resize(test, dsize=(256, 256), interpolation=cv2.INTER_AREA)
#cv2.imshow("test", test_rs)

#print(test_rs.shape)

img_lst = []
#for img in apple_lst:
#    img_lst.append(cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)) # 이미지 크기 변환

def plot_cluster(arr):
    n = len(arr)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < n:
                img = arr[idx]
                ax[i, j].imshow(img)
            ax[i, j].axis('off')
    plt.show()

plot_cluster(img_lst)

