# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:16:10 2017

@author: charlie
"""

### K Means clustering algorithm 
### Test dataset is the self-made 2D dataset or Iris dataset 


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

### Initialize the dataset of x, y and label
x   = np.append(np.arange(1, 20), np.arange(41, 60))
y   = np.append(np.arange(2, 21), 1.5 * np.arange(45, 64))
x   = np.append(x, np.arange(71, 90))
y   = np.append(y, 2.5 * np.arange(81, 100))

### Add noise
x2  = x + 10 * np.random.random(len(x)) + 10 * np.random.random(len(x))
y2  = y + 10 * np.random.random(len(y)) + 10 * np.random.random(len(x))
x   = np.append(x, x2)
y   = np.append(y, y2)

#### Import Iris data
#iris = datasets.load_iris()
#data_reduced_2d = PCA(n_components = 2).fit_transform(iris.data)
#
#### Use raw data 
##x = iris.data[:, 0]
##y = iris.data[:, 1]
#
#### Use decomposited data
#x = data_reduced_2d[:, 0]
#y = data_reduced_2d[:, 1]

len_x = len(x)
label_init = np.append(np.zeros(len_x / 2),  np.ones(len_x / 2))
label_c = np.zeros(len_x)

#dataset = np.append(np.append(x, y), label_init)
#dataset = dataset.reshape(3, len_x)

num_clusters = 2
#x_centroid = np.random.rand(min(x), max(x), num_clusters)
#y_centroid = np.random.rand(min(y), max(y), num_clusters)
x_centroid = (max(x) - min(x)) * np.random.random(num_clusters) + min(x)
y_centroid = (max(y) - min(y)) * np.random.random(num_clusters) + min(y)
x_centroid_temp = max(x) * np.random.random(num_clusters) + min(x)
y_centroid_temp = max(y) * np.random.random(num_clusters) + min(y)


fig = plt.figure()
plt.ion()
plt.scatter(x, y, c = label_c)
plt.scatter(x_centroid, y_centroid, c = 'k')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data plot')
plt.pause(1)


max_delta_c = 1
criteria = 1e-3
num_iterations = 0
while(max_delta_c > criteria):
    ### The numbers of points which belong to each centroid point change every iteration
    count_c = np.zeros(num_clusters)
    
    ### Get the closest centroids of each [x[i], y[i]]
    for i in range(0, len_x):
        x_c_dist = x_centroid - x[i]
        y_c_dist = y_centroid - y[i]
        c_dist = np.sqrt(x_c_dist**2 + y_c_dist**2)
        index = np.where(c_dist == min(c_dist))
        label_c[i] = index[0]        
        count_c[int(label_c[i])] += 1
        
    ### Update new x_centroid and y_centroid
    for j in range(0, num_clusters):
#        print x[np.where(label_c == j)]
        x_centroid_temp[j] = np.mean(x[np.where(label_c == j)])
        y_centroid_temp[j] = np.mean(y[np.where(label_c == j)])
    
    ### Calculate the difference between the old and new centroid, then compare
    ### with the convergence criteria. If max_delta_c < criteria, then stop 
    ### iteration and we get the final clustering scheme.
    max_delta_c_x = max(abs(x_centroid - x_centroid_temp))
    max_delta_c_y = max(abs(y_centroid - y_centroid_temp))
    max_delta_c = max(max_delta_c_x, max_delta_c_y)
    
    fig.clear()
    plt.scatter(x, y, c = label_c)
    plt.scatter(x_centroid, y_centroid, c = 'r' )
    plt.pause(1)
    
    
    ### Update the x and y of each centroid point
    x_centroid = x_centroid_temp
    y_centroid = y_centroid_temp
    
    num_iterations += 1 
        
#plt.figure()
#plt.scatter(x, y, c = label_c)
#plt.scatter(x_centroid, y_centroid, c = 'r' )
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Data plot')
#plt.show()
        
        
        
        
        
        
        
        
        
        
        
