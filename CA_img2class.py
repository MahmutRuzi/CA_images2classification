#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:11:00 2023

@author: Mahmut Ruzi
"""

# import libraries

import numpy as np
from scipy.stats import randint, loguniform
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, LearningCurveDisplay
from sklearn.metrics import  accuracy_score, classification_report, ConfusionMatrixDisplay

from collections import Counter

rng = np.random.RandomState(5971325)
#%%
# load data
raw_data = np.load('labeled_CA_data.npz')

ca_imgs = raw_data["images"]
lab_class = raw_data["labels"]

#%%
# explore the data
n_samples = len(ca_imgs)
img_size = ca_imgs.shape
print(Counter(lab_class))

#%%
# a help function to visualize multiple images
def plot_images(X, labels, title):
    """function to plot large number of CA images."""
    num_imgs = X.shape[0] 
    nrows = int(np.sqrt(num_imgs))
    ncols = int(np.ceil(num_imgs/nrows))
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    
    for img, ax, label in zip(X, axs.ravel(), labels):
        ax.imshow(img.reshape((32, 32)), cmap="gray", interpolation="None")
        ax.axis("off")
        ax.set_title("%i" % label)
    fig.suptitle(title, fontsize=12)

#%%

plot_images(ca_imgs, lab_class,'Contact angle imges')

#%%
# flatten images, and scale

data = ca_imgs.reshape((n_samples, -1))/255

# split the data into 80% train and 20% test subset
X_train, X_test, y_train, y_test = train_test_split(data, lab_class, 
                                                    test_size=0.2, shuffle=True,
                                                    random_state=rng,
                                                    stratify=lab_class)
#%%
# create SVM classifier, scan for best parameters

param_dist = {
    'kernel': ['linear','poly','rbf'],
    "C": loguniform(0.1, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}

estimator_svc = svm.SVC(class_weight='balanced', decision_function_shape='ovr')
n_param_samples = 1000 
clf1 = RandomizedSearchCV(estimator_svc, param_dist, n_iter=n_param_samples,
                          scoring='accuracy', n_jobs=8, cv=5,
                          random_state=rng)
#%%
svc_estimator = clf1.fit(X_train, y_train)

print("Best estimator found by grid search:")
print(f"best parameters: {svc_estimator.best_params_}")
print(f"Mean cross-validated score: {svc_estimator.best_score_:.2f}")

#%%
# learn the images using the best parameters
# initiate the svm classifier
svc_best = svm.SVC(C=10, kernel="rbf", gamma=0.0007, class_weight='balanced',
              random_state=rng)
# fit the classfier using the training datatset
svc_best.fit(X_train, y_train)
# predict
predicted = svc_best.predict(X_test)

svc_accuracy = accuracy_score(y_test, predicted)
print("Accuracy (train) for SVC: %0.1f%% " % (svc_accuracy * 100))

#%%

print(
    f"Classification report for classifier {svc_best}:\n"
    f"{classification_report(y_test, predicted)}\n"
)

#%%
disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted,normalize='true')
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#%%
# DecisionTree classifier

n_param_samples2 = 100

# Define the parameter grid to search
param_dist = {
    "max_depth": [2, 3, 4, 5, 6, 7],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "criterion": ["gini", "entropy"]
}

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=rng,class_weight='balanced', 
                             max_features='log2')

# Initialize the RandomizedSearchCV object
dt_tune = RandomizedSearchCV(clf, param_distributions=param_dist, 
                                   n_iter=n_param_samples2, cv=5, verbose=False,
                                   random_state=rng, scoring='accuracy')
#%%
# Fit the model on the training data
dt_tune.fit(X_train, y_train)

#%%
# Print the best parameters
print(f"Mean cross-validated score: {dt_tune.best_score_:.2f}")
print(dt_tune.best_params_)

#%%
# initiate a DT classifier using the best parameters
dt_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=7, 
                                      min_samples_leaf=3, min_samples_split=8,
                                      max_features='log2', random_state=rng, 
                                      class_weight='balanced')

#%%
# train the DT classifier
dt_estimator.fit(X_train, y_train)
#%%
# make prediction
y_pred_dt = dt_estimator.predict(X_test)
#%%
# prediction accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Accuracy (train) for DecisionTree: %0.1f%% " % (dt_accuracy * 100))

#%%
# Make learning plot

train_sizes = [5, 10, 20, 30, 40, 60, 79]
#%%
# Plot learning curve
LearningCurveDisplay.from_estimator(DecisionTreeClassifier(criterion='entropy',
                                                           max_depth=6, 
                                                           min_samples_split=8,
                                                           min_samples_leaf=1, 
                                                           max_features='log2', 
                                                           class_weight='balanced', 
                                                           random_state=rng),
                                    data, lab_class, train_sizes=train_sizes, 
                                    cv=5)


#%%
#xGB classify
# Setting the params

params = {"gamma": [0, 0.01, 0.1, 0.25, 1.0],"learning_rate": [0.01, 0.05, 0.1, 0.3], 
          "max_depth": [3,4,5,6], "colsample_bytree": [0.3,0.5, 1],
          "min_child_weight": [0, 1, 3, 5, 10]}

#%%

xgb_clf_tun = xgb.XGBClassifier(booster='gbtree',tree_method='hist',
                                objective="multisoftprob:logistic", num_class=3,
                                seed=42, early_stopping_rounds=10, 
                                eval_metric="mlogloss")
n_param_samples2 = 100 
tuned_params = RandomizedSearchCV(xgb_clf_tun, params, 
                            verbose=0, n_jobs=20, cv=5, n_iter=n_param_samples2, 
                            random_state=rng, scoring='accuracy') 

#%%

tuned_params.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)], verbose=False) 

#%%
# Printing the best parameters

print(tuned_params.best_params_)
#%%
# initiate the xgboost classifier using the best parameters
ev_metrics = ["mlogloss", "merror", "auc"] # Various metric parameters for later visualzation

clf_xgb_boosted = xgb.XGBClassifier(booster='gbtree',
                                    objective="multisoftprob:logistic",
                                    num_class=3, seed=42, gamma = 0.0, 
                                    learning_rate=0.3, max_depth = 4,
                                    colsample_bytree =0.5, early_stopping_rounds=10,
                                    min_child_weight=0, 
                                    eval_metric=ev_metrics)

#%%
clf_xgb_boosted.fit(X_train, y_train, eval_set=[(X_test, y_test)])

#%%
# Model prediction

y_pred_xgb_boosted = clf_xgb_boosted.predict(X_test)

# probablity of prediction

y_pred_xgb_prob = clf_xgb_boosted.predict_proba(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb_boosted )
print("Accuracy (train) for xgboost: %0.1f%% " % (xgb_accuracy * 100))

#%%
# Plotting the confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb_boosted ,normalize='true')
disp.figure_.suptitle("xgb Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
#%%
#plot training epoch 
# retrieve training rate
results = clf_xgb_boosted.evals_result()

#%%
training_rate = results['validation_0']
epochs = len(training_rate[ev_metrics[0]])
x_axis = range(0, epochs)

num_sub_plots =len(ev_metrics)
#%%

fig, axs = plt.subplots(nrows=num_sub_plots, ncols=1, figsize=(8, 12))

for i in range(num_sub_plots):
    axs[i].plot(x_axis, training_rate[ev_metrics[i]],'k-o')
    axs[i].set_ylabel(ev_metrics[i], fontsize=12)

axs[i].set_xlabel('epoch')

#%%

