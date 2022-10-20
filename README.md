# CNNAI: A Convolution Neural Network-Based Latent Fingerprint Matching Using the Combination oNearest Neighbor Arrangement Indexing
By Uttam U. Deshpande et al.




For a CNN based accurate fingerprint segmentation, minutiae extraction, and matching refer to this paper: https://www.frontiersin.org/articles/10.3389/frobt.2020.00113/full

## Introduction
We present a local minutia-based Convolution Neural Network (CNN) based matching model called “Combination of Nearest Neighbor Arrangement Indexing (CNNAI).” This model makes use of a set of “n” local nearest minutiae neighbor features and generates rotation-scale invariant feature vectors. 

## CNNAI ALGORITHM
For latent ﬁngerprint matching, we use the nearest combination of minutiae points around a central minutia. We obtain the discriminative invariants based on the minutiae structures and store them on the hash-table for matching. To make the matcher robust against scale, rotation, and missing minutiae, we deﬁne the triangular minutiae structure. 
![image](https://user-images.githubusercontent.com/107185323/196980826-1c24fa65-dfe8-47af-a7ed-d2ec2c80f1db.png)

The arrangement vectors of query ﬁngerprints are compared against the stored vectors. The voting method is used to increase the vote count for matching minutiae belonging to a particular ﬁngerprint. The arrangement vector count of diﬀerent ﬁngerprints is sorted in decrement order, and the count with the highest voting is chosen as Rank-1 retrieved ﬁngerprint.
![image](https://user-images.githubusercontent.com/107185323/196980939-beae49a3-d72a-4843-86e1-d183170b963a.png)

The proposed matching model employs neural network techniques for classifying a query latent ﬁngerprint from a class of a given set of pre-trained classes depending upon the arrangement vectors. One-dimensional convolutional layer is used in designing the matching model. 
![image](https://user-images.githubusercontent.com/107185323/196982908-aabe5cc1-49d2-48f2-99a5-ab822209889d.png)





