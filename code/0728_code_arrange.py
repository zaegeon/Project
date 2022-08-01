# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Version Check
# python --version ⇒ Python 3.9.10
print('np version :', np.__version__) # np version : 1.23.1
print('cv2 version :', cv2.__version__) # cv2 version : 4.6.0
import pkg_resources
print('matplotlib version :', pkg_resources.get_distribution('matplotlib').version) # matplotlib version : 3.5.2
print('sklearn version :', pkg_resources.get_distribution('sklearn').version)

# File Open
file_path = 'D:/JaeGeon/# 2. Study/! Project/apple' # apple image file path
path = './'
os.chdir(file_path) # Apple Image가 있는 폴더로 경로 변경
file_lst = os.listdir(path)

# Create Image List
apple_lst = []
for file in file_lst:
    apple_lst.append(plt.imread(file))

# Plot Function (이미지 시각화)
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

# X, y train-test-split

# train, test txt file
# import random as rd
#
# 일반(0)과 병든 사과(1) 각각 500개들 중 80%를 train set, 20%를 test set으로 설정 & txt 파일로 저장
# all_num = list(np.arange(1, 501))
# train_num = rd.sample(all_num, int(len(all_num) * 0.8))
# test_num = list(set(all_num).difference(train_num))
# f = open('train.txt', 'w') # train.txt 생성
# for i in range(0,2):
#     for n in train_num:
#         data = path + 'apple_' + str(i) +'_' + format(n, '03') + '.jpg\n'
#         f.write(data)
# f.close()
#
# f = open('test.txt', 'w') # test.txt 생성
# for i in range(0,2):
#     for n in test_num:
#         data = path + 'apple_' + str(i) +'_' + format(n, '03') + '.jpg\n'
#         f.write(data)
# f.close()

os.chdir('D:/JaeGeon/# 2. Study/! Project') # train.txt 파일이 있는 곳으로 경로 변경
# file_path = 'D:/JaeGeon/# 2. Study/! Project/apple' # apple image file path
X_tr = []
with open('train.txt', mode='rt') as f:
    for line in f:
        l = line.strip()
        train_num = file_path + '/' + l[-15:]
        X_tr.append(cv2.imread(train_num))
X_tr = np.asarray(X_tr)

X_te = []
with open('test.txt', mode='rt') as f:
    for line in f:
        l = line.strip()
        test_num = file_path + '/' + l[-15:]
        X_te.append(cv2.imread(test_num))
X_te = np.asarray(X_te)

y_tr = []
with open('train.txt', mode='rt') as f:
    for line in f:
        a = line.strip()
        y_tr.append(int(a[-9:-8]))
y_tr = np.asarray(y_tr)

y_te = []
with open('test.txt', mode='rt') as f:
    for line in f:
        a = line.strip()
        y_te.append(int(a[-9:-8]))
y_te = np.asarray(y_te)

print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape) # shape check

# Scaling (0 ~ 255 → [0, 1) int)
X_tr = X_tr / 255
X_te = X_te / 255

# Reshape for analysis
X_tr_rs = X_tr.reshape(800, -1)
X_te_rs = X_te.reshape(200, -1)
print(X_tr_rs.shape, X_te_rs.shape)

# KNN : 계산 시 시간이 오래 걸림
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(X_tr_rs, y_tr)
print('=== KNeighborsClassifier n_neighbors = 5 (default) ===')
print('KNN Score (Train) :', knn.score(X_tr_rs, y_tr)) # train score = 0.70
print('KNN Score (Test) :', knn.score(X_te_rs, y_te)) # test score = 0.635

# KNN plot - neighbors 개수에 따른 score 시각화
tr_score = []
te_score = []
neighbors = range(1, 11)
for x in neighbors:
    knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=x)
    knn.fit(X_tr_rs, y_tr)
    tr_score.append(knn.score(X_tr_rs, y_tr))
    te_score.append(knn.score(X_te_rs, y_te))

plt.plot(neighbors, tr_score, label='Train Score')
plt.plot(neighbors, te_score, label='Test Score')
for i in range(len(tr_score)):
    plt.text(neighbors[i], ( tr_score[i] + te_score[i] )/ 2 , f'Diff : \n{round((tr_score[i] - te_score[i]), 4)}', color='green')
plt.show()
# n=4 ⇒ low diff & high accuracy ∴ best

knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
knn.fit(X_tr_rs, y_tr)
print('=== KNeighbors Classifier n_neighbors = 4 (best) ===')
print('KNN Score (Train) :', knn.score(X_tr_rs, y_tr)) # train score = 0.785
print('KNN Score (Test) :', knn.score(X_te_rs, y_te)) # test score = 0.69

# SGD Classifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss', max_iter=30)
sgd.fit(X_tr_rs, y_tr)
print('=== SGD Classifier ===')
print('SGD Score (Train) :', sgd.score(X_tr_rs, y_tr)) # train score = 1.0
print('SGD Score (Test) :', sgd.score(X_te_rs, y_te)) # test score = 0.92

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(n_jobs=-1, max_iter=30)
lr.fit(X_tr_rs, y_tr)
print('=== Logistic Regression ===')
print('Logistic Regression Score (Train) :', lr.score(X_tr_rs, y_tr))
print('Logistic Regression Score (Test) :', lr.score(X_te_rs, y_te))

# PCA
from sklearn.decomposition import PCA
pca_rgb = []
for i in X_tr:
    b, g, r = cv2.split(i)
    r = r / 255
    g = g / 255
    b = b / 255
    pca_r = PCA(n_components=32)
    pca_r_tran = pca_r.fit_transform(r)
    pca_g = PCA(n_components=32)
    pca_g_tran = pca_g.fit_transform(g)
    pca_b = PCA(n_components=32)
    pca_b_tran = pca_b.fit_transform(b)

    pca_r_org = pca_r.inverse_transform(pca_r_tran)
    pca_g_org = pca_g.inverse_transform(pca_g_tran)
    pca_b_org = pca_b.inverse_transform(pca_b_tran)

    pca_img = cv2.merge((pca_r_org, pca_g_org, pca_b_org))
    pca_rgb.append(pca_img)

from sklearn.decomposition import PCA
pca_rgb_te = []
for i in X_te:
    b, g, r = cv2.split(i)
    r = r / 255
    g = g / 255
    b = b / 255
    pca_r = PCA(n_components=32)
    pca_r_tran = pca_r.fit_transform(r)
    pca_g = PCA(n_components=32)
    pca_g_tran = pca_g.fit_transform(g)
    pca_b = PCA(n_components=32)
    pca_b_tran = pca_b.fit_transform(b)

    pca_r_org = pca_r.inverse_transform(pca_r_tran)
    pca_g_org = pca_g.inverse_transform(pca_g_tran)
    pca_b_org = pca_b.inverse_transform(pca_b_tran)

    pca_img = cv2.merge((pca_r_org, pca_g_org, pca_b_org))
    pca_rgb_te.append(pca_img)

pca_rgb_tr = np.asarray(pca_rgb)
pca_rgb_tr = pca_rgb_tr.reshape((800, -1))
pca_rgb_te = np.asarray(pca_rgb_te)
pca_rgb_te = pca_rgb_te.reshape((200, -1))
print(pca_rgb_tr.shape, pca_rgb_te.shape)

knn.fit(pca_rgb_tr, y_tr)
print('=== KNeighbors Classifier, n_neighbors = 4, PCA O ===')
print('KNN Score (Train) :', knn.score(pca_rgb_tr, y_tr))
print('KNN Score (Test) :', knn.score(pca_rgb_te, y_te))

sgd = SGDClassifier(max_iter=1000)
sgd.fit(pca_rgb_tr, y_tr)
print('=== SGD Classifier, PCA O ===')
print('SGD Score (Train) :', sgd.score(pca_rgb_tr, y_tr))
print('SGD Score (Test) :', sgd.score(pca_rgb_te, y_te))

lr.fit(pca_rgb_tr, y_tr)
print('=== Logistic Regression, PCA O ===')
print('Logistic Regression Score (Train) :', lr.score(pca_rgb_tr, y_tr))
print('Logistic Regression Score (Test) :', lr.score(pca_rgb_te, y_te))
