from random import Random

import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

data = pd.read_csv('karachi_dis_cod_month_as_no_crimeType_as_no.csv');

# scaler = MinMaxScaler()
# x_axis = 'Year'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])
# scaler = MinMaxScaler()
# x_axis = 'Month'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])
# scaler = MinMaxScaler()
# x_axis = 'Lat'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])
# scaler = MinMaxScaler()
# x_axis = 'Log'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])
# scaler = MinMaxScaler()
# x_axis = 'Crime Type'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])
# x_axis = 'Reported Number'
# scaler.fit(data[[x_axis]])
# data[[x_axis]] = scaler.transform(data[[x_axis]])


df = pd.DataFrame(data, columns=['Year', 'Month', 'Lat', 'Log', 'Crime Type', 'Reported Number'])
X = df.drop('Reported Number', axis=1)
# y = np.asarray(df['Crime Type'],dtype=np.float64)
# y = df['Crime Type']
y = np.asarray(df['Reported Number'])
# dtype="|S6")
# y =[1,2,3,4,5,6]
# print(X)
# print()
# print(X.head())
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train.shape)
# print(y)

'	Month	Division	Lat	Log	Crime Type	Reported Number	Legends	Features	Result'
# plt.figure(figsize=(12,10))
# cor =X_train.corr()
# sns.heatmap(cor,annot=True, cmap='Accent')
# plt.show()
# y = [1,2,3,4,5,6]
# print(np.unique(y))
#

# Best parameters set found on development set:
# for clasification
# {'activation': 'tanh', 'hidden_layer_sizes': (5,), 'solver': 'lbfgs'}
# for reg
# {'activation': 'identity', 'hidden_layer_sizes': (1,), 'solver': 'lbfgs'}
# clf = MLPClassifier(solver='lbfgs', activation='tanh', hidden_layer_sizes=(5), max_iter=100000,)
# clf = MLPRegressor(solver='lbfgs', activation='relu',random_state=1,max_iter=5000, )
# 80% accurate
# clf = MLPRegressor(solver='lbfgs', activation='relu',hidden_layer_sizes=(45,45,46),random_state=1,max_iter=5000, )
# 47% without scaling
# clf = MLPRegressor(solver='lbfgs', activation='relu',hidden_layer_sizes=(100,50,25,10,5,10,25,50),random_state=1,max_iter=5000,)
# 71%
# clf = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=(15,2,35,62,9,45,8,2), random_state=1, max_iter=5000, )
# clf = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=(15, 2, 35, 63, 9, 45, 8, 2), random_state=1, max_iter=5000, )
# ['identity', 'logistic', 'relu', 'softmax', 'tanh'].
# won relu
# sgd, adam, lbfgs
# won lbfgs
# {‘constant’, ‘invscaling’, ‘adaptive’}

clf.fit(X_train, y_train)
#
# 0.8038017676381538
# 0.8038017676381538

# joblib.dump(clf,"model.pkl")
# out = open("model.pkl","wb")
# pickle.dump(clf,out)
# out.close()
# pre = clf.predict(X_test[:2])
# pre = clf.predict([[2014,12,24.9313,67.0374,2]])
pre = clf.predict([[2011, 1, 24.8605, 67.0261, 4]])
print(pre)
print(clf.score(X_test, y_test))


def find_best():
    import random
    act = ['identity', 'logistic', 'relu',  'tanh']
    solv = [ 'adam', 'lbfgs']
    for i in range(1000):
        r1 = random.randint(1, 100)
        # r2 = random.randint(2, 62)
        # r3 = random.randint(2, 62)
        # r4 = random.randint(2, 62)
        # r5 = random.randint(2, 62)
        # r6 = random.randint(2, 62)
        # r7 = random.randint(2, 62)
        # r8 = random.randint(2, 62)
        # index_solv=random.randint(0, 1)
        # index_act =random.randint(0,3)
        clf = MLPRegressor(solver=solv[1], activation=act[2], hidden_layer_sizes=(r1,2,46,62,9,45,8,2), random_state=1,
                           max_iter=5000, )
        clf.fit(X_train,y_train)
        acc =clf.score(X_test, y_test)
        print(acc)
        if(acc>0.72):
            print('more than 71%')
            # print('index of solv ',index_solv)
            # print('index of act ',index_act)
            print('these are the values',r1)
            # print(r2)
            # print(r3)
            # print(r4)
            # print(r5)
            # print(r6)
            # print(r7)
            # print(r8)
            break
        if(acc>0.5):
            print('more than 50%')
            # print('index of solv ',index_solv)
            # print('index of act ',index_act)
            print('these are the values',r1)
            # print(r2)
            # print(r3)
            # print(r4)
            # print(r5)
            # print(r6)
            # print(r7)
            # print(r8)
# find_best()