import pandas as pd
import numpy as np
# import random as rd
import matplotlib.pyplot as plt

data = pd.read_csv('karachi_dis_cod_month_as_no.csv')
# print(data.head())
x_axis = "Month"
y_axis = "Reported Number"
X = data[[x_axis, y_axis]]
# Visualise data points
# plt.scatter(X[x_axis],X[y_axis],c='black')
# plt.xlabel(x_axis)
# plt.ylabel('Reported')
# plt.show()

K = 8

# Select random observation as centroids
Centroids = (X.sample(n=K))
# plt.scatter(X[x_axis],X[y_axis],c='black')
# plt.scatter(Centroids[x_axis],Centroids[y_axis],c='red')
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.show()

diff = 1
j = 0

while diff != 0:
    XD = X
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c[x_axis] - row_d[x_axis]) ** 2
            d2 = (row_c[y_axis] - row_d[y_axis]) ** 2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i = i + 1

    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i + 1] < min_dist:
                min_dist = row[i + 1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C
    Centroids_new = X.groupby(["Cluster"]).mean()[[y_axis, x_axis]]
    if j == 0:
        diff = 1
        j = j + 1
    else:
        diff = (Centroids_new[y_axis] - Centroids[y_axis]).sum() + (Centroids_new[x_axis] - Centroids[x_axis]).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[[y_axis, x_axis]]
color = ['blue', 'green', 'cyan', 'yellow', 'purple', 'gray', 'orange', 'brown', 'silver']
for k in range(K):
    data = X[X["Cluster"] == k + 1]
    plt.scatter(data[x_axis], data[y_axis], c=color[k])
plt.scatter(Centroids[x_axis], Centroids[y_axis], c='red')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.show()
print(data)
