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

##########################################################################

os.chdir('C:/Users/itwill/Desktop/apple')
path = "./"
file_lst = os.listdir(path)

apple_lst = []
for file in file_lst:
    apple_lst.append(plt.imread(file))

X = np.array(apple_lst.copy())
y = np.concatenate([np.zeros(500), np.ones(500)])
print(np.unique(y, return_counts=True))

plot_cluster(X[0:100]) # 시각화

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=0.2, stratify=y)

print(X_tr.shape, X_te.shape)
print(y_tr.shape, y_te.shape)

X_tr = X_tr.reshape((800, -1))
X_te = X_te.reshape((800, -1))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

knn=KNeighborsClassifier(n_jobs=-1)
knn.fit(X_tr, y_tr)

print(knn.score(X_tr, y_tr))
