import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import pickle
def new_model(csv):
    df= DataFrame(csv, columns=['Year', 'Month', 'Lat', 'Log', 'Crime Type', 'Reported Number'])
    X,y = df.drop('Reported Number', axis=1), np.asarray(df['Reported Number'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=(15, 2, 35, 62, 9, 45, 8, 2),
                       random_state=1, max_iter=2000, )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    out = open("user_model.pkl","wb")
    pickle.dump(clf,out)
    out.close()
    return acc
