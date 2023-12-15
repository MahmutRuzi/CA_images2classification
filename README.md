# CA_images2classification
This repo contains data,python code to explore supervised machine learning algorisms for classification. I have tried the following supervised learning models: support vector machines, decision tree, and gradient boosting. Besides doing basic classification task using optimized hyperparameters (using RandomizedSearchCV), the python code includes many helpful routines to visualize the results. 

The data comprised of 99 pictures and provided as a numpy array, showing water contact angle (CA) on a flat surface. Such pictures are commonly used for measuring the contact angle of liquid droplets on a flat surface. Each picture is 32 by 32 pixels, and belongs to one the three categories: superhydrophobic (CA>=150 degrees, label=2), hydrophobic (90 < CA < 150 degrees, label=1), and hydrophilic (CA<90 degrees, label=0). 

Before running the python code, please make sure you setup proper environment. I use conda to create and install packages, and use Spyder as a IDE. The environment I use is provided in the yml file, you might not need all of them. But the following is necessary.
1. Use conda to install sklearn, matplotlib, numpy, and scipy
2. Use pip to install xgboost: pip install xgboost
