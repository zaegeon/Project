# imports
import numpy as np
import matplotlib.pyplot as plt
import os

# file open
# os.chdir('D:/JaeGeon/# 2. Study/! Project/apple')
os.chdir('C:/Users/itwill/Desktop/apple')
path = "./"
file_lst = os.listdir(path)

apple_lst = []
for file in file_lst:
    apple_lst.append(plt.imread(file))

# X, y
X = np.array(apple_lst.copy())
X_sc = X / 255 # Scaling
y = np.concatenate([np.zeros(500), np.ones(500)])

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

# train-test split
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_sc,y, test_size=0.2, stratify=y)
X_tr = X_tr.reshape((800, -1))
X_te = X_te.reshape((200, -1))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_jobs=-1)
knn.fit(X_tr, y_tr)
print(knn.score(X_tr, y_tr)) # train score : 0.7375
print(knn.score(X_te, y_te)) # test score : 0.64

# Wrong Prediction plot
def wrong_pred(X_tr, y_tr):
    tr_pred = knn.predict(X_tr)
    X_wrong = X_tr[tr_pred != y_tr]
    y_wrong = y_tr[tr_pred != y_tr]
    wrong_pred = tr_pred[tr_pred != y_tr]

    fig, ax = plt.subplots(5, 5, figsize = (8, 8), constrained_layout=True)
    for i in range(5):     # subplot의 row index를 0~9까지 반복
        for j in range(5): # subplot의 column index를 0~9까지 반복
            n = len(X_wrong)
            randidx = np.random.choice(n, 25, replace=False)
            for idx in randidx:
                img = X_wrong[idx].reshape((256, 256, 3))
                ax[i, j].imshow(img, cmap='binary')
                ax[i, j].axis('off')
                if y_wrong[idx] == 0:
                    ax[i, j].set_title('N / D')
                else:
                    ax[i, j].set_title('D / N')

    plt.show()

wrong_pred(X_tr, y_tr)

# confusion matrix
from sklearn.metrics import confusion_matrix
tr_pred = knn.predict(X_tr)
cm = confusion_matrix(y_tr, tr_pred)
print(cm)
knn_precision = cm[0,0] / (cm[0,0] + cm[1,0])
print(knn_precision) # Precision : 정상 예측 중 실제 정상인 것의 비율

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

# Validation
# os.chdir('D:/JaeGeon/# 2. Study/! Project/apple_val')
os.chdir('C:/Users/itwill/Desktop/apple_val')
path = "./"
file_lst2 = os.listdir(path)

apple_val_lst = []
for file in file_lst2:
    apple_val_lst.append(plt.imread(file))

# X, y
X_val = np.array(apple_val_lst.copy())
X_val_sc = X_val / 255 # Scaling
y_val = np.concatenate([np.zeros(86), np.ones(99)])
print(np.unique(y_val, return_counts=True))

# print(X_val_sc.shape)
X_val_sc = X_val_sc.reshape(185, -1)
val_pred = sgd.predict(X_val_sc)

print(np.mean(y_val == val_pred)) # 0.87

# Validation X, y
X_val = np.array(apple_val_lst.copy())
X_val_sc = X_val / 255 # Scaling
y_val = np.concatenate([np.zeros(86), np.ones(99)])
print(np.unique(y_val, return_counts=True))

# print(X_val_sc.shape)
X_val_sc = X_val_sc.reshape(185, -1)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
pca.fit(X_tr)
X_tr_pca = pca.transform(X_tr)
X_te_pca = pca.transform(X_te)
# print(X_tr_pca.shape)

knn.fit(X_tr_pca, y_tr)
print(knn.score(X_tr_pca, y_tr))

# PCA - KNN Pipeline
from sklearn.pipeline import Pipeline
knn = Pipeline([('pca', PCA(n_components=32)),
                ('knn_clf', KNeighborsClassifier(n_neighbors=4))])
pca_r = pca.fit_transform(r_scaled)
pca_g = pca.fit_transform(g_scaled)
pca_b = pca.fit_transform(b_scaled)
img_compressed = cv2.merge((pca_r.inverse_transform(pca_r), pca_g.inverse_transform(pca_g), pca_b.inverse_transform(pca_b)))

knn.fit(img_compressed, y_tr)
print('KNN Score (Train) :', knn.score(img_compressed, y_tr))
print('KNN Score (Test) :', knn.score(X_te, y_te))

knn_pred = knn.predict(X_val_sc)
print('KNN Score (Validation) :', round(np.mean(y_val == knn_pred), 4))
print('====================================================')

# PCA - SGD Pipeline
sgd = Pipeline([('pca', PCA(n_components=32)),
                ('sgd', SGDClassifier(n_jobs=-1, loss='log_loss', max_iter=1000, tol=None))])
sgd.fit(X_tr, y_tr)
print('SGD Score (Train) :', sgd.score(X_tr, y_tr))
print('SGD Score (Test):', sgd.score(X_te, y_te))

sgd_pred = sgd.predict(X_val_sc)
print('SGD Score (Validation) :', round(np.mean(y_val == sgd_pred), 4))
print('====================================================')

# PCA - LR Pipeline
from sklearn.linear_model import LogisticRegression
logistic = Pipeline([('pca', PCA(n_components=32)),
                     ('logistic', LogisticRegression(n_jobs=-1, max_iter=100))])
logistic.fit(X_tr, y_tr)
print('LogisticRegression Score (Train) :', logistic.score(X_tr, y_tr))
print('LogisticRegression Score (Test) :', logistic.score(X_te, y_te))

log_pred = logistic.predict(X_val_sc)
print('LogisticRegression Score (Validation) :', round(np.mean(y_val == log_pred), 4))
print('====================================================')

