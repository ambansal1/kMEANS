import os
import csv
import numpy as np
import kmeans
import scipy

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')
C = np.zeros((7,2))

a = kmeans.update_assignments(X,C)

#print "C before update"
#print C
C = kmeans.update_centers(X,C,a)
#print "C after update"
#print C

[C,a] = kmeans.lloyd_iteration(X,C)
#print "c after convergence"
#print C

obj = kmeans.kmeans_obj(X, C, a)
#print obj

# TODO: Test update_assignments function, defined in kmeans.py

# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py

# TODO: Test kmeans_obj function, defined in kmeans.py

# TODO: Run experiments outlined in HW6 PDF
print "checking size of x"
'''

bestobj1 = np.zeros((20))
for temp in range(1,21):
    [_,_,bestobj] = kmeans.kmeans_cluster(X, temp , 'fixed' , 10)
    print temp
    bestobj1[temp-1] = bestobj



plt.plot(bestobj1,'ro')
plt.ylabel('objective function value')
plt.show()
'''

### EXPERIMENT 7
bestobj2 = np.zeros((1000))
for temp in range(0,1000):
    print temp
    [_,_,bestobj] = kmeans.kmeans_cluster(X, 9 , 'random' , 1)

    bestobj2[temp] = bestobj

print "mean"
print  (np.sum(bestobj2))/1000



# For question 9 and 10
# from sklearn.decomposition import PCA
mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
