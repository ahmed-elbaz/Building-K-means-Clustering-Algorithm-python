# Building-K-means-Clustering-Algorithm-python

## Introduction ##

In this project we will implement the K-means(k=3) clustering algorithm then apply it to the data provided in the file “Data.txt”. Each row in the file corresponds to one data point. One important aspect of K-means that changes the results significantly is the initialization. the strategy used for initializing cluster centers is as follows:  

1- We pick one of the dataset points randomly as the center of the first cluster.  
2- For the next cluster, we find the point with maximum distance to the center of the previous cluster.  
3- Choose this point as the center of the next cluster.  
4- Repeat steps 2 and 3 to initialize the center of the third cluster under one condition that center of third cluster can't be the same as first one.  

K-means works perfectly with spherical data but the accuracy decreases If the shape of data differs and as we said initialization significantly affects clustering result so we will run the K-means algorithm with the initialization method 100 times.

The final output of the K-means clustering is the result that gives the minimum average distance between the points and the centers of their corresponding clusters.

## Repository contents ##

The repository includes   

1. k-means.py file this is the code for the whole project.  
2. Kmeans.jpg image it is a plot that shows the data points after clustering in addition to the centroids of each cluster.  
3. Data.txt file and it inludes the data points that we need to cluster.  
