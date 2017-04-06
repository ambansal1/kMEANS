import numpy as np
import math
import os
########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def update_assignments(X, C):

  a = np.zeros(X.shape[0])

  difference_matrix= np.sqrt(np.sum(np.square(X[:, np.newaxis] - C), axis=2))

  a = np.argmin(difference_matrix, axis=1)

  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array

  return a

def update_centers(X, C, a):


  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  max_clusterindex = np.max(a) + 1
  sumOfCluster = np.zeros(((max_clusterindex),2))
  countOfCluster = np.zeros((max_clusterindex))


  number = 0
  for number in range(0,a.shape[0]):

    sumOfCluster[a[number]] = sumOfCluster[a[number]] + X[number,:]
    countOfCluster[a[number]] =  countOfCluster[a[number]] + 1

  #countOfCluster[countOfCluster == 0] = ['1']

  for count in range(0,C.shape[0]):
    if(countOfCluster[count] == 0):
      sumOfCluster[count] = C[count]
      countOfCluster[count] = 1

  C = (sumOfCluster.T)/(countOfCluster.T)
  C = C.T





  #Output:
  #   C is the new cluster centers (k, d), 2-d array

  return C



def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array

  a = update_assignments(X, C)
  aprevious = np.zeros((a.shape[0]))

  while(np.sum(np.square((aprevious - a)))):


    aprevious = a
    C = update_centers(X, C, a)
    a = update_assignments(X, C)




  return (C, a)

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  obj = 0.0
  for index in range(0, X.shape[0]):
    obj = obj + np.sum(np.square(X[index,:] - C[a[index]]))
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float


  return obj


########################################################################
#######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
########################################################################

# kmeans_cluster will be used in the experiments, it is available after you 
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)



########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix