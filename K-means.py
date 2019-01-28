# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:51:19 2019

@author: Baz-PC
"""

###############
#Used libraries
###############

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance


#################
#Reading the data
#################

os.chdir("E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 4")
data = x=np.loadtxt("Data.txt")

####################################
#Plotting the data before clustering
####################################

#As data points are 2 dimensional we will plot the data and we can see that data is spherical so K-means
# clustering should work perfectly when k=3

plt.scatter(data[:,0],data[:,1],s=5) # ploting X1 vs X2
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data(X1,X2) before clustering')
plt.show()

##########################################
#Getting initial centers of the 3 clusters
##########################################

#Calcualting euclidean distances between all points and put the value in dist list
dist=euclidean_distances(data)

#Center list will contain 100 values of the 3 cluster centers
Center=[]

for i in range(100):
    # Choosing center of first cluster randomly
    InitC1Index=np.random.choice(data.shape[0])
    InitC1=data[InitC1Index,:]
    
    # Getting second cluster center's index and value
    InitC2Index=np.argmax(dist[InitC1Index,:])
    InitC2=data[InitC2Index,:]
    
    # Getting third cluster center's index and value and we will put a condition to avoid geting the third
    #center the same as second or first center which means we will have 2 points instead of 3
    if np.argmax(dist[InitC2Index,:]) != InitC1Index:
        InitC3Index=np.argmax(dist[InitC2Index,:])
    else:
        #to avoid having Miu1=Miu3 we will choose the second highest maximum distance(index=1) as index=0
        #is the maximum distance equals initial center 1
        InitC3Index=np.argsort(dist[InitC2Index,:])[1]
    InitC3=data[InitC3Index,:]
     
    #append centers value for each interation to InitC
    Center.append([InitC1,InitC2,InitC3])

##########################################
# Mean function for each column in 2D list
##########################################
def average_column(data):
    
    sumCol1=0
    sumCol2=0
    
    for point in range(len(data)):
        sumCol1=sumCol1+data[point][0]
        sumCol2=sumCol2+data[point][1]
    
    return [sumCol1/len(data),sumCol2/len(data)]
    
##########################
#K-means with k=3 function
##########################

#There are 4 inputs for this function the data itself in addition to the initial centers for the three
#clusters. The function will iterate till Mius of the 3 centers in one iteration is the same like last
#iteration. this function returns a list of 6 variables the three centers after convergence and 
#the points in each cluster
    
def kmeans(data,Miu1Init,Miu2Init,Miu3Init):

    #the below variables will be used to compare Miu of each cluster for certain iteration to Miu of this
    #cluster in the previous iterationiteration If distance between them is 0 for all clusters so 
    #algorithm converges
        
    errorC1=1
    errorC2=1
    errorC3=1
    
    while (errorC1+errorC2+errorC3!=0):
        
        #rnk will take values of 1,2 and 3 based on which cluster this point belongs in certain iteration
        rnk=[]
    
        for point in range(len(data)):
          
            #compute the distance between point and centers
            distC1=distance.euclidean(data[point],Miu1Init)
            distC2=distance.euclidean(data[point],Miu2Init)
            distC3=distance.euclidean(data[point],Miu3Init)
            
            #Compute rnk 
            if (distC1<=distC2) & (distC1<=distC3):
                rnk.append(1)
            elif (distC2<distC1) & (distC2<=distC3):
                rnk.append(2)
            else:
                rnk.append(3) 
        
        #We separate the points that belong to each cluster in 3 variables
        cluster1=data[[i for i, e in enumerate(rnk) if e == 1]]  
        cluster2=data[[i for i, e in enumerate(rnk) if e == 2]]
        cluster3=data[[i for i, e in enumerate(rnk) if e == 3]]
            
        #We get updated Mius based on rnk
        Miu1=average_column(cluster1)
        Miu2=average_column(cluster2)
        Miu3=average_column(cluster3)
        
        errorC1=distance.euclidean(Miu1,Miu1Init)
        errorC2=distance.euclidean(Miu2,Miu2Init)
        errorC3=distance.euclidean(Miu3,Miu3Init)
       
        if (errorC1+errorC2+errorC3!=0):
            Miu1Init=Miu1
            Miu2Init=Miu2
            Miu3Init=Miu3
    
    return [Miu1,Miu2,Miu3,cluster1,cluster2,cluster3]


###########################################################################
#Applying kmeans function 100 times for each intial centers selected before
###########################################################################

# We will define 6 variables three of them are for centers in all iteration. for example Center1 will 
#100 values of centers for the different iterations. Also we will define 3 variables for the points 
#that belong to each center
Center1=[]
Center2=[]
Center3=[]
points1=[]
points2=[]
points3=[]

for i in range(100):
    output=kmeans(data,Center[i][0],Center[i][1],Center[i][2])
    Center1.append(output[0])
    Center2.append(output[1])
    Center3.append(output[2])
    points1.append(output[3])
    points2.append(output[4])
    points3.append(output[5])

########################################################################################################
#Getting the minimum average distance between the points and the centers of their corresponding clusters
########################################################################################################

#list that will contain the average distance between each point and center for each iteration
Distance_Iteration=[]

for iteration in range(100):
    
    # variables to increment the distance betwwen each point and its cluster center
    distance1=0
    distance2=0
    distance3=0
    
    for point in range(100):
        dist1=distance.euclidean(Center1[iteration],points1[iteration][point])
        dist2=distance.euclidean(Center2[iteration],points2[iteration][point])
        dist3=distance.euclidean(Center3[iteration],points3[iteration][point])
        
        distance1+=dist1
        distance2+=dist2
        distance3+=dist3
        
        Average_distance=(distance1/len(points1[0]))+(distance2/len(points2[0]))+(distance3/len(points3[0]))
    
    #Distance iteration list will include 100 values for average distance for each iteration
    Distance_Iteration.append(Average_distance)

#This variable includes the index of best iteration
Best_Iteration=np.argmin(Distance_Iteration)
print("Iteration with the minimum average distance between points and centers is iteration number "+str(Best_Iteration+1))
print("Centers of the best iteration are "+str(Center1[Best_Iteration])+" and "+str(Center2[Best_Iteration])+" and "+str(Center3[Best_Iteration]))  

###################################
#Plotting the data after clustering
###################################
#This function will set a differnet color for points in one cluster

def color_condition(data,Best_Iteration):
    color_condition=[]
    
    for point in range(len(data)):
        if [data[point,0],data[point,1]] in points1[Best_Iteration]:
            color='blue'
        elif [data[point,0],data[point,1]] in points2[Best_Iteration]:
            color='green'
        else:
            color='red'
        
        color_condition.append(color)  
    return color_condition

#plotting points in the same cluster with same color in addition to plotting centers
plt.scatter(data[:,0],data[:,1],s=9,alpha=0.2,c=color_condition(data,Best_Iteration)) # ploting X1 vs X2
plt.scatter(Center1[Best_Iteration][0],Center1[Best_Iteration][1],s=50,marker='^',c='blue',label='Cluster 1 Center')
plt.scatter(Center2[Best_Iteration][0],Center2[Best_Iteration][1],s=50,marker='^',c='green',label='Cluster 2 Center')
plt.scatter(Center3[Best_Iteration][0],Center3[Best_Iteration][1],s=50,marker='^',c='red',label='Cluster 3 Center')
plt.legend(loc='upper left')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data(X1,X2) after clustering')
plt.savefig('Kmeans.jpg')

