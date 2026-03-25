# Homework 1 - PCA 


## Part 2

### 2.1 Centered Data

`[[-1. -3. -4.]
 [-1. -1. -2.]
 [ 1.  1.  2.]
 [ 1.  3.  4.]]`

 ### 2.2 Covariance

 Covariance matrix :
 `[[ 1.33  2.67  4.  ]
 [ 2.67  6.67  9.33]
 [ 4.    9.33 13.33]]`

The covariance matrix is as expected symmetric.
Along the diagonal, we can see the variance of each variable. The variance of the last parameter is clearly the largest. The values not on the diagonal represent the covariances between the different parameters. We can see that the values tend to behave in a similar way and that the covariance with the third parameter is strong. 

### 2.3 Eigenvalues


`[21.08  0.25  0.  ]`

### 2.4 Projection 

The data is reduced in dimensionality using Principal Component Analysis (PCA). First, the data is centered by subtracting the mean of each variable. Then, the covariance matrix is computed and decomposed into eigenvalues and eigenvectors. The eigenvector associated with the largest eigenvalue defines the first principal component, which represents the direction of maximum variance.

In this case, the largest eigenvalue is 21.08, showing that most of the variance lies along this single direction. To reduce the data, each observation is projected onto this first principal component by taking the dot product with its eigenvector. This transforms the original 3-dimensional data into a 1-dimensional representation.

The resulting values (e.g., 5.09, 2.39, -2.39, -5.09) are the coordinates of the data along the main axis, preserving most of the information while reducing complexity.


## 2.5 Interpretation

The eigenvalues of the covariance matrix indicate how the total variance is distributed across different directions in the data. Large eigenvalues correspond to directions where the data varies a lot, while small eigenvalues indicate directions with little to no variation.

In your case, the eigenvalues are approximately 21.08, 0.25, and 0. This shows that almost all the variance is concentrated in the first principal component. The presence of very small or zero eigenvalues implies that some variables are highly redundant, meaning they can be almost entirely explained by a combination of the others.

This also suggests that the intrinsic dimensionality of the data is much lower than the original dimension. Although the data is in 3D, it essentially lies close to a 1D subspace, since only one eigenvalue is significant.

Finally, in terms of variance explained, the first eigenvalue accounts for nearly 100% of the total variance. This means that projecting the data onto the first principal component retains almost all the important information, while the remaining components contribute very little.
