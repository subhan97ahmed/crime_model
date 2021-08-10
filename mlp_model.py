import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

data = pd.read_csv('karachi_dis_cod_month_as_no_crimeType_as_no.csv')

df = pd.DataFrame(data, columns=['Year', 'Month', 'Lat', 'Log', 'Crime Type', 'Reported Number'])
X = df.drop('Reported Number', axis=1)
y = np.asarray(df['Reported Number'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# '	Month	Division	Lat	Log	Crime Type	Reported Number	Legends	Features	Result'

# 80% accurate
# clf = MLPRegressor(solver='lbfgs', activation='relu',hidden_layer_sizes=(45,45,46),random_state=1,max_iter=5000, )
# 71% 15,2,35,62,9,45,8,2
clf = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=(15,2,35,62,9,45,8,2), random_state=1, max_iter=5000,)
clf.fit(X_train, y_train)

# out = open("model.pkl","wb")
# pickle.dump(clf,out)
# out.close()
# pre = clf.predict(X_test[:2])
# pre = clf.predict([[2011, 1, 24.8605, 67.0261, 4]])
# print(pre)
print(clf.score(X_test, y_test))

