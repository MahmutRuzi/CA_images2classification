# CA_images2classification
This repo contains data (raw CA pictures and the corresponding labels), colab notebook, and python code to explore supervised machine learning algorisms for a classification task. I have tried the following supervised learning models: support vector machines, decision tree, and gradient boosting. Besides doing basic classification task using optimized hyperparameters (using RandomizedSearchCV), the python code and colab notebook includes many helpful routines to visualize the results. 

The data comprised of 99 pictures and provided as a numpy array, showing water contact angle (CA) on a flat surface. Such pictures are commonly used for measuring the contact angle of a liquid droplet on a flat surface. Each picture is 32 by 32 pixels gray scale, and belongs to one of the three categories: superhydrophobic (CA>=150 degrees, label=2), hydrophobic (90 < CA < 150 degrees, label=1), and hydrophilic (CA<90 degrees, label=0). 

You can use either the colab notebook (recommended) or the python code. Using the colab notebook is easy, just download open the notebook, and upload the CA pictures data as detailed in the notebook, then run, which takes less than two minutes on free version of the colab. 

If you want to use the python code, please make sure you setup proper environment. I use conda to create and install packages, and use Spyder as the IDE. The environment I used is provided in the yml file, you might not need all of them. But the following is necessary.
1. Use conda to install sklearn, matplotlib, numpy, and scipy
2. Use pip to install xgboost: pip install xgboost

Enjoy ! 
