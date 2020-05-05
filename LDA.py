# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:58:26 2020

@author: Hiro
"""

import numpy as np
##lets use the iris dataset for linear discriminant analysis
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

names = iris.target_names

#compute the mean vectors
mean_vectors = [X[y==i].mean(axis=0) for i in range(len(names))]

total_mean = X.mean(axis=0)

#compute the scatter matrices
S_B = np.zeros((X.shape[1], X.shape[1]))

for i in range(len(names)):
    n = X[y==i].shape[0]
    dot = np.matmul((mean_vectors[i].reshape((4,1)) - total_mean.reshape((4,1))),(mean_vectors[i].reshape((4,1)) - total_mean.reshape((4,1))).T)
    S_B += n*dot
    
#within class scatter matrix
S_W = np.zeros((X.shape[1], X.shape[1]))

for i in range(len(names)):
    
    S_i = np.zeros_like(S_W)
    
    for j in range(len(X[y==i])):
        S_i += np.matmul((X[y==i][j].reshape((4,1)) - mean_vectors[i].reshape((4,1))),(X[y==i][j].reshape((4,1)) - mean_vectors[i].reshape((4,1))).T )
    
#     S_i = np.sum(np.matmul((X[y==i][j].reshape((4,1)) - mean_vectors[i].reshape((4,1))),(X[y==i][j].reshape((4,1)) - mean_vectors[i].reshape((4,1))).T )for j in range(len(X[y==i])))
    S_W += S_i
    
#find eigenvalues and eigenvectors of scatter matrices
from numpy.linalg import eigh, inv

eig_vals, eig_vects = eigh(np.matmul(inv(S_W), S_B))

#if you want to flip the order so it is in descending order
eig_vals = eig_vals[::-1]
eig_vects = eig_vects[:,::-1]

eig_sum = np.sum(eig_vals)
cum_sums = [np.sum(eig_vals[:i+1])/eig_sum for i in range(len(eig_vals))]

tol = .9

num_vects_keep = 1
i=0

while cum_sums[i] < tol:
    num_vects_keep +=1
    i+=1
    
W = eig_vects[:,:num_vects_keep]

T = np.matmul(X, W)