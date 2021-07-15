import pandas as pd
import numpy as np
# import random as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

data = pd.read_csv('karachi_dis_cod_month_as_no_crimeType_as_no.csv')
# print(data.head())
x_axis = "Crime Type"
y_axis = "Reported Number"
z_axis = "Month"
# X = data[[x_axis, y_axis, z_axis]]
# print(X)

# preprocessing
scaler = MinMaxScaler()
scaler.fit(data[[x_axis]])
data[[x_axis]] = scaler.transform(data[[x_axis]])
# for y_axis
scaler = MinMaxScaler()
scaler.fit(data[[y_axis]])
data[[y_axis]] = scaler.transform(data[[y_axis]])
# for z_axis
scaler = MinMaxScaler()
scaler.fit(data[[z_axis]])
data[[z_axis]] = scaler.transform(data[[z_axis]])
X = data[[x_axis, y_axis, z_axis]]

# Visualise data points
fig = plt.figure()
ax = plt.axes(projection='3d')
zdata = X[z_axis]
xdata = X[x_axis]
ydata = X[y_axis]

km = KMeans(n_clusters=4)
y_predict = km.fit_predict(X[[x_axis, y_axis, z_axis]])
print(y_predict)
X['cluster'] = y_predict

print(X['cluster'])
df1 = X[X.cluster == 0]
df2 = X[X.cluster == 1]
df3 = X[X.cluster == 2]
df4 = X[X.cluster == 3]

ax.scatter3D(df1[x_axis], df1[y_axis], df1[z_axis], c='g', marker='o')
ax.scatter3D(df2[x_axis], df2[y_axis], df2[z_axis], c='purple', marker='^')
ax.scatter3D(df3[x_axis], df3[y_axis], df3[z_axis], c='b', marker='H')
ax.scatter3D(df4[x_axis], df4[y_axis], df4[z_axis], c='y', marker='s')
ax.scatter3D(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],km.cluster_centers_[:, 2], c='r',marker='*')
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_zlabel(z_axis)

plt.show()

# # print(X)
# K = 4
#
# # selecting random observation as centroids
# Centroids = (X.sample(n=K))
# # plt.scatter(X[x_axis],X[y_axis],c='black')
# # plt.scatter(Centroids[x_axis],Centroids[y_axis],c='red')
# # plt.xlabel(x_axis)
# # plt.ylabel(y_axis)
# # plt.show()
#
# diff = 1
# j = 0
#
# while diff != 0:
#     XD = X
#     i = 1
#     for index1, row_c in Centroids.iterrows():
#         ED = []
#         for index2, row_d in XD.iterrows():
#             d1 = (row_c[x_axis] - row_d[x_axis]) ** 2
#             d2 = (row_c[y_axis] - row_d[y_axis]) ** 2
#             d = np.sqrt(d1 + d2)
#             ED.append(d)
#         X[i] = ED
#         i = i + 1
#
#     C = []
#     for index, row in X.iterrows():
#         min_dist = row[1]
#         pos = 1
#         for i in range(K):
#             if row[i + 1] < min_dist:
#                 min_dist = row[i + 1]
#                 pos = i + 1
#         C.append(pos)
#     X["Cluster"] = C
#     Centroids_new = X.groupby(["Cluster"]).mean()[[y_axis, x_axis]]
#     if j == 0:
#         diff = 1
#         j = j + 1
#     else:
#         diff = (Centroids_new[y_axis] - Centroids[y_axis]).sum() + (Centroids_new[x_axis] - Centroids[x_axis]).sum()
#         print(diff.sum())
#     Centroids = X.groupby(["Cluster"]).mean()[[y_axis, x_axis]]
# color = ['blue', 'green', 'cyan', 'yellow', 'purple', 'gray', 'orange', 'brown', 'silver']
# for k in range(K):
#     data = X[X["Cluster"] == k + 1]
#     plt.scatter(data[x_axis], data[y_axis], c=color[k])
# plt.scatter(Centroids[x_axis], Centroids[y_axis], c='red')
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.show()
# # print(data)
