# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:37:47 2017

@author: charlie
"""
### Code of 'Python for data anaylis' 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()



## Plot 2D graph of sepal length
x = iris.data[:, 0]     # X - Axis - sepal length
y = iris.data[:, 1]     # Y - Axis - sepal length
species = iris.target
# Scatter plot

plt.figure()
plt.title('Iris dataset - Classification by sepal sizes')
plt.scatter(x, y, c = species)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


### Plot 2D graph of petal length
x = iris.data[:, 0]     # X - Axis - petal length
y = iris.data[:, 1]     # Y - Axis - petal length
species = iris.target
# Scatter plot

plt.figure()
plt.title('Iris dataset - Classification by petal sizes')
plt.scatter(x, y, c = species)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
#
#plt.xticks(())
#plt.yticks(())
plt.show()


### Plot 2D graph after decomposition with PCA methods
data_reduced_2d = PCA(n_components = 2).fit_transform(iris.data)

plt.figure()
plt.scatter(data_reduced_2d[:,0], data_reduced_2d[:,1], c=species)
plt.xlabel('First eigenvector')
plt.ylabel('Second eigenvector')
plt.title('2D Iris dataset by PCA')
plt.show()


### Plot 3D graph after decomposition with PCA methods
data_reduced = PCA(n_components = 3).fit_transform(iris.data)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('3D Iris dataset by PCA')
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')

ax.scatter(data_reduced[:,0], data_reduced[:,1], data_reduced[:,2], c = species)

plt.show()




