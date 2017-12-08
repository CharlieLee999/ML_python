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
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#num_clusters =  #input('Please input the number of clusters:')

### ******************
### mak_blobs
### Creat datasets with sklearn.datasets.samples_generator.make_blobs function
#X, y_true = datasets.samples_generator.make_blobs(n_samples = 300, \
#                centers = num_clusters, cluster_std = 0.5, random_state = 0)
#
#x = X[:, 0]
#y = X[:, 1]

### *****************
###  load_digits()
digits = datasets.load_digits()
labels = digits.target
digits_data = scale(digits.data)
n_samples, n_feature = digits_data.shape
n_digits = len(np.unique(labels))
reduced_data = PCA(n_components = 2).fit_transform(digits_data)

kmeans = KMeans( n_clusters = n_digits, n_init = 10) #init = 'k-means++',
X = reduced_data
num_clusters = n_digits
x = X[:, 0]
y = X[:, 1]
### Use the sklearn.cluster.KMeans function to cluster data
#kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans)
plt.scatter(centers[:, 0], centers[:, 1], c ='k')
plt.show()

### *****************
### Initialize the dataset of x, y and label
#x   = np.append(np.arange(1, 20), np.arange(41, 60))
#y   = np.append(np.arange(2, 21), 1.5 * np.arange(45, 64))
#x   = np.append(x, np.arange(71, 90))
#y   = np.append(y, 2.5 * np.arange(81, 100))

### ****************
##### Add noise
#x2  = x #+ 20 * np.random.random(len(x)) + 20 * np.random.random(len(x))
#y2  = y + 20 * np.random.random(len(y)) + 20 * np.random.random(len(x))
#x3  = x
#y3  = 2 * y + 40 * np.random.random(len(x)) 

#x   = np.append(x, x2)
#y   = np.append(y, y2)
#x   = np.append(x, x3)
#y   = np.append(y, y3)


### **************
###  Datasets of Exercise 6.2 
#x = [1,1, 2,2,2, 3,3, 4, 6, 8, 9,9]
#y = [9,3, 7,2,1, 8,2, 1, 5, 7, 6,4]
#
#x = np.asarray(x)
#y = np.asarray(y)

### *************
#### Import Iris data
#iris = datasets.load_iris()
#data_reduced_2d = PCA(n_components = 2).fit_transform(iris.data)
#
### ******************
#### Use raw data 
##x = iris.data[:, 2]
##y = iris.data[:, 3]
#tag = iris.target
#### Use decomposited data
#x = data_reduced_2d[:, 0]
#y = data_reduced_2d[:, 1]

len_x = len(x)
#label_init = np.append(np.zeros(len_x / 2),  np.ones(len_x / 2))
label_c = np.zeros(len_x)

#num_clusters = 4 #int(input('Please input the number of clusters: '))
x_centroid = (max(x) - min(x)) * np.random.random(num_clusters) + min(x)
y_centroid = (max(y) - min(y)) * np.random.random(num_clusters) + min(y)
x_centroid_temp = np.zeros(len(x_centroid)) #max(x) * np.random.random(num_clusters) + min(x)
y_centroid_temp = np.zeros(len(y_centroid)) #max(y) * np.random.random(num_clusters) + min(y)

print ('x_centroid and y_centroid are', x_centroid, y_centroid)
print ('x_centroid_temp and y_centroid_temp are', x_centroid_temp, y_centroid_temp)
print ('\n')


fig = plt.figure()
plt.ion()
plt.scatter(x, y, c = label_c)
plt.scatter(x_centroid, y_centroid, c = 'k')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.title('Data plot')
plt.pause(1)


max_delta_c = 1
criteria = 1e-9
num_iterations = 0
while( max_delta_c > criteria ):  # max_delta_c > criteria  #num_iterations < 20
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
        if count_c[j] == 0:
            pass
        else:
            x_centroid_temp[j] = np.mean(x[np.where(label_c == j)])
            y_centroid_temp[j] = np.mean(y[np.where(label_c == j)])
    print ('Iteration', num_iterations)
    print ('x_centroid and y_centroid are', x_centroid, y_centroid)
    print ('x_centroid_temp and y_centroid_temp are', x_centroid_temp, y_centroid_temp)
    
    
    ### Calculate the difference between the old and new centroid, then compare
    ### with the convergence criteria. If max_delta_c < criteria, then stop 
    ### iteration and we get the final clustering scheme.
    max_delta_c_x = max(abs(x_centroid - x_centroid_temp))
    max_delta_c_y = max(abs(y_centroid - y_centroid_temp))
    max_delta_c = max(max_delta_c_x, max_delta_c_y)
    print ('max_delta_c is ', max_delta_c)    
    
    fig.clear()
    plt.scatter(x, y, c = label_c)
    plt.scatter(x_centroid, y_centroid, c = 'r' )
    plt.pause(1)
    
    
    ### Update the x and y of each centroid point
    x_centroid = x_centroid_temp.copy()
    y_centroid = y_centroid_temp.copy()
    print ('x_centroid and y_centroid are', x_centroid, y_centroid)
    print ('\n')
    num_iterations += 1 

plt.savefig('K_Means_clustering')







#plt.figure()
#plt.scatter(x, y, c = tag)
#plt.scatter(x_centroid, y_centroid, c = 'r' )
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Data plot')
#plt.show()
        
        
        
        
        
        
        
        
        
        
        
