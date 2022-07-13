# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# file open
os.chdir('C:/Users/itwill/Desktop/apple')
path = "./"
file_lst = os.listdir(path)
#print(file_lst)

# Create Image List
apple_lst = []
for file in file_lst:
    apple_lst.append(plt.imread(file))

# X, y
X = np.array(apple_lst.copy())
X_sc = X / 255 # Scaling
y = np.concatenate([np.zeros(500), np.ones(500)])
print(np.unique(y, return_counts=True))

# plot function
def plot_cluster(arr, labels):
    n = len(arr)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10), constrained_layout=True)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < n:
                img = arr[idx]
                ax[i, j].imshow(img)
            ax[i, j].axis('off')
            if labels[idx] == 0:
                ax[i, j].set_title('Normal')
            elif labels[idx] == 1:
                ax[i, j].set_title('Disease')
    plt.show()

plot_cluster(X[450:550], y[450:550])

ridx = np.random.choice(1000, 100)
plot_cluster(X[ridx], y[ridx])

# train-test split
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_sc,y, test_size=0.2, stratify=y)

print(X_tr.shape, X_te.shape)
print(y_tr.shape, y_te.shape)

# reshape (800, 256, 256, 3) → (800, 196608(256x256x3))
X_tr = X_tr.reshape((800, -1))
X_te = X_te.reshape((200, -1))

print(X_tr.shape, X_te.shape)
print(y_tr.shape, y_te.shape)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_jobs=-1)
knn.fit(X_tr, y_tr)

print(knn.score(X_tr, y_tr)) # 0.745
print(knn.score(X_te, y_te)) # 0.725

# Wrong Prediction plot
def wrong_pred(X_tr, y_tr):
    tr_pred = knn.predict(X_tr)
    X_wrong = X_tr[tr_pred != y_tr]
    y_wrong = y_tr[tr_pred != y_tr]
    wrong_pred = tr_pred[tr_pred != y_tr]

    fig, ax = plt.subplots(10, 10, figsize = (10, 10), constrained_layout=True)
    for i in range(10):     # subplot의 row index를 0~9까지 반복
        for j in range(10): # subplot의 column index를 0~9까지 반복
            img = X_wrong[i * 10 + j].reshape((256, 256, 3))
            ax[i, j].imshow(img, cmap='binary')
            ax[i, j].axis('off')
            if y_wrong[i * 10 + j] == 0:
                ax[i, j].set_title('N / D')
            else:
                ax[i, j].set_title('D / N')

    plt.show()

wrong_pred(X_tr, y_tr)

# KNN - adjust n_neighbors
tr_score = []
te_score = []
neighbors = range(1, 11)
for x in neighbors:
    knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=x)
    knn.fit(X_tr, y_tr)
    tr_score.append(knn.score(X_tr, y_tr))
    te_score.append(knn.score(X_te, y_te))

plt.plot(neighbors, tr_score, label='Train Score')
plt.plot(neighbors, te_score, label='Test Score')
for i in range(len(tr_score)):
    plt.text(neighbors[i], ( tr_score[i] + te_score[i] )/ 2 , f'Diff : \n{round((tr_score[i] - te_score[i]), 4)}', color='green')
plt.show()

# n_neighbors = 4
knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=4)
knn.fit(X_tr, y_tr)
print(knn.score(X_tr, y_tr))
wrong_pred(X_tr, y_tr)

# confusion matrix
from sklearn.metrics import confusion_matrix
tr_pred = knn.predict(X_tr)
cm = confusion_matrix(y_tr, tr_pred)
print(cm)
knn_precision = cm[0,0] / (cm[0,0] + cm[1,0])
print(knn_precision) # 278 / (278+11) = 0.962

# SGD Classifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss', max_iter=1000)
sgd.fit(X_tr, y_tr)
print(sgd.score(X_tr, y_tr))
print(sgd.score(X_te, y_te))

tr_score = []
te_score = []
for i in range(1, 31):
    sgd = SGDClassifier(loss='log_loss', max_iter=i, tol=None)
    sgd.fit(X_tr, y_tr)
    tr_score.append(sgd.score(X_tr, y_tr))
    te_score.append(sgd.score(X_te, y_te))

    plt.plot(tr_score, label='Train Score')
    plt.plot(te_score, label='Test Score')
    plt.show()

# epoch = 26 → Good Score

# Validation
os.chdir('C:/Users/itwill/Desktop/apple_val')
path = "./"
file_lst2 = os.listdir(path)

apple_val_lst = []
for file in file_lst2:
    apple_val_lst.append(plt.imread(file))

# X, y
X_val = np.array(apple_val_lst.copy())
X_val = X_val / 255 # Scaling
y_val = np.concatenate([np.zeros(100), np.ones(100)])
print(np.unique(y_val, return_counts=True))

sgd.fit(X_val, y_val)
val_pred = sgd.predict(X_val)
sgd.score(X_val, y_val)