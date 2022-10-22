# CNNAI: A Convolution Neural Network-Based Latent Fingerprint Matching Using the Combination oNearest Neighbor Arrangement Indexing
By Uttam U. Deshpande et al.




For a CNN based accurate fingerprint segmentation, minutiae extraction, and matching refer to this paper: https://www.frontiersin.org/articles/10.3389/frobt.2020.00113/full

## Introduction
We present a local minutia-based Convolution Neural Network (CNN) based matching model called “Combination of Nearest Neighbor Arrangement Indexing (CNNAI).” This model makes use of a set of “n” local nearest minutiae neighbor features and generates rotation-scale invariant feature vectors. 

### CNNAI ALGORITHM
For latent ﬁngerprint matching, we use the nearest combination of minutiae points around a central minutia. We obtain the discriminative invariants based on the minutiae structures and store them on the hash-table for matching. To make the matcher robust against scale, rotation, and missing minutiae, we deﬁne the triangular minutiae structure. 
![image](https://user-images.githubusercontent.com/107185323/196980826-1c24fa65-dfe8-47af-a7ed-d2ec2c80f1db.png)

The arrangement vectors of query ﬁngerprints are compared against the stored vectors. The voting method is used to increase the vote count for matching minutiae belonging to a particular ﬁngerprint. The arrangement vector count of diﬀerent ﬁngerprints is sorted in decrement order, and the count with the highest voting is chosen as Rank-1 retrieved ﬁngerprint.
![image](https://user-images.githubusercontent.com/107185323/196980939-beae49a3-d72a-4843-86e1-d183170b963a.png)

The proposed matching model employs neural network techniques for classifying a query latent ﬁngerprint from a class of a given set of pre-trained classes depending upon the arrangement vectors. One-dimensional convolutional layer is used in designing the matching model. 
![image](https://user-images.githubusercontent.com/107185323/196982908-aabe5cc1-49d2-48f2-99a5-ab822209889d.png)

The repository includes:
* Source code of CoarseNet.
* Training code 
* Pre-trained weights 
* Jupyter notebooks

### Citing
@ARTICLE{10.3389/frobt.2020.00113,
AUTHOR={Deshpande, Uttam U. and Malemath, V. S. and Patil, Shivanand M. and Chaugule, Sushma V.},    
TITLE={CNNAI: A Convolution Neural Network-Based Latent Fingerprint Matching Using the Combination of Nearest Neighbor Arrangement Indexing},      
JOURNAL={Frontiers in Robotics and AI},      
VOLUME={7},           
YEAR={2020},        
URL={https://www.frontiersin.org/articles/10.3389/frobt.2020.00113},      	
DOI={10.3389/frobt.2020.00113},      	
ISSN={2296-9144}, 
}

## Requirements: software
`Python 2.7` `Tensorflow 1.7.0` `Keras 2.1.6`

## Installation 
`conda install cv2, numpy, scipy, matplotlib, pydot, graphviz`

Download models and put into `Models` folder.
* CoarseNet: [Googledrive](https://drive.google.com/file/d/1bU3T-XQRlKy6C77e5eD-DOD_QlNlAIjR/view?usp=sharing)
* FineNet: [Googledrive](https://drive.google.com/file/d/1rQw6hs-3hv_7WqJQ8ZYhJhi4laa-9qbY/view?usp=sharing)
* CNNAI FVC 2002: [Googledrive](https://drive.google.com/file/d/18XtX_U3IDTwsRCZS-QF65XG3zN_E_-sN/view?usp=sharing)
* CNNAI FVC 2004: [Googledrive](https://drive.google.com/file/d/1TIuIuuuIenjGg3adIWrDsomi9o-JZfAS/view?usp=sharing)
* CNNAI NIST SD27: [Googledrive](https://drive.google.com/file/d/1S5SOOpc671vRey-Z9Xgj9aH2GScHCbSx/view?usp=sharing)

## Testing Procedure
* Based on which fingerprint to test, copy respective cnai_weight.h5 file present in CNNAI FVC 2002, CNNAI FVC 2004 or CNNAI NIST SD27 in Models folder alongwith CoarseNet.h5 and FineNet.h5 files.
* Save images to be tested in `/Dataset/CoarseNet_test/` folder.
* To perform minutiae extraction and matching, Run `main_work.py` present in CoarseNet folder to observe the matching results. Or You can run the notebook `main_work_NIST.ipynb` present in CoarseNet folder.
* Observe output in `output_CoarseNet` folder.
