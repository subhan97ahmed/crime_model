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
km = KMeans(n_clusters=5)
y_predict = km.fit_predict(X[[x_axis, y_axis, z_axis]])
print(y_predict)
X['cluster'] = y_predict

print(X['cluster'])
df1 = X[X.cluster == 0]
df2 = X[X.cluster == 1]
df3 = X[X.cluster == 2]
df4 = X[X.cluster == 3]
df5 = X[X.cluster == 4]

ax.scatter3D(df1[x_axis], df1[y_axis], df1[z_axis], c='g', marker='o', label='cluster 1')
ax.scatter3D(df2[x_axis], df2[y_axis], df2[z_axis], c='purple', marker='^', label='cluster 2')
ax.scatter3D(df3[x_axis], df3[y_axis], df3[z_axis], c='b', marker='H', label='cluster 3')
ax.scatter3D(df4[x_axis], df4[y_axis], df4[z_axis], c='y', marker='s', label='cluster 4')
ax.scatter3D(df5[x_axis], df5[y_axis], df5[z_axis], c='black', marker='|', label='cluster 5')
ax.scatter3D(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],km.cluster_centers_[:, 2], c='r', marker='*', label='centroid')
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_zlabel(z_axis)
ax.legend()
plt.show()
