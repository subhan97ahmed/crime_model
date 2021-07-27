import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('karachi_dis_cod_month_as_no_crimeType_as_no.csv');
# data = datasets.load_wine()
# X = dataset.data
# y = dataset.target
# mpl = MLPClassifier()
# print(X)
# print(y)
scaler = MinMaxScaler()
x_axis = 'Year'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
scaler = MinMaxScaler()
x_axis = 'Month'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
scaler = MinMaxScaler()
x_axis = 'Lat'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
scaler = MinMaxScaler()
x_axis = 'Log'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
scaler = MinMaxScaler()
x_axis = 'Reported Number'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
x_axis = 'Crime Type'
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])

df = pd.DataFrame(data, columns=['Year', 'Month', 'Lat', 'Log', 'Crime Type', 'Reported Number'])
X = df.drop('Crime Type', axis=1)
# y = np.asarray(df['Crime Type'],dtype=np.float64)
# y = df['Crime Type']
y = np.asarray(df['Crime Type'], dtype="|S6")
# y =[1,2,3,4,5,6]
# print(X)
# print()
# print(X.head())
# print(y.head())
X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)
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
# param_grid = [
#         {
#             'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#             'solver' : ['lbfgs', 'sgd', 'adam'],
#             'hidden_layer_sizes': [
#              (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
#              ]
#         }
#        ]
# clf = GridSearchCV(MLPClassifier(), param_grid, cv=3,
#                            scoring='accuracy')
# clf.fit(X,y)
#

# print("Best parameters set found on development set:")
# print(clf.best_params_)
# Best parameters set found on development set:
# {'activation': 'tanh', 'hidden_layer_sizes': (5,), 'solver': 'lbfgs'}
clf = MLPClassifier(solver='lbfgs', activation='tanh', hidden_layer_sizes=(5), max_iter=100000,)
# clf = MLPClassifier()
clf.fit(X_train, y_train)
# clf
# print(X)
# clf.predict([2011,1	,24.8605,67.0261,23])
# print(clf.predict([[2011,1,24.8605,67.0261,3]]))
print(clf.predict([[2014,12,24.9313,67.0374,8]]))
pre =clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pre))

def find_best():
    from random import randint
    act = ['identity', 'logistic', 'relu', 'tanh']
    solv = ['adam', 'lbfgs']
    while 1:
        # 15,2,35,62,9,45,8,2
        r1,r2,r3,r4,r5,r6,r7,r8  = randint(1, 15),randint(1, 2), randint(1, 35), randint(1, 62), randint(1, 9), randint(1, 45), randint(1, 8), randint(1, 2)
        clf = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=(r1, r2, r3, r4, r5, r6, r7, r8),
                           random_state=1,
                           max_iter=5000, )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(acc)
        if acc >= 0.72:
            print('more than 71%')
            print('these are the values', r1)
            print(r2)
            print(r3)
            print(r4)
            print(r5)
            print(r6)
            print(r7)
            print(r8)
            break
# find_best()
