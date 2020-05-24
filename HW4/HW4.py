import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score


# ## Load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


# 550 data with 300 features
print(x_train.shape)

# It's a binary classification problem 
print(np.unique(y_train))


# ## Question 1
# Implement the K-fold cross-validation function. Your function should take K as an argument and return a list (len(list) should equal to K), contains K elements, each element is a list contains the training fold and validation fold

def cross_validation(x_train, y_train, k=5):
    return NotImplementedError

kfold_data = cross_validation(x_train, y_train, k=10)
assert len(kfold_data) == 10 # should contain 10 fold of data
assert len(kfold_data[0]) == 2 # each element should contain train fold and validation fold
assert len(kfold_data[0][0].shape[0]) == 90 # The number of data in each fold should equal to training data divieded by K


# ## Question 2
# Using sklearn.svm.SVC to train a classifier on the provided train set and conduct the grid search of “C”, “kernel” and “gamma” to find the best parameters by cross-validation.

clf = SVC(C=1.0, kernel='rbf', gamma=0.01)

## your code


print(best_parameters)


# ## Question 3
# Plot the grid search results of your SVM. The x, y, z represent the parameters of “C”, “kernel” and “gamma”, respectively. And the color represents the average score of validation folds.
# You reults should be same as the reference image ![image](https://plotly.com/~xoelop/147.png) 

# ## Question 4
# Train your SVM model by the best parameters you found from question 2 on the whole training set and evaluate the performance on the test set. **You accuracy should over 0.8**


y_pred = best_model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))


# ## Question 5
# Compare the performance of each model you have implemented from HW1 to HW3


train_df = pd.read_csv("../HW1/train_data.csv")
x_train = train_df['x_train'].to_numpy().reshape(-1,1)
y_train = train_df['y_train'].to_numpy().reshape(-1,1)

test_df = pd.read_csv("../HW1/test_data.csv")
x_test = test_df['x_test'].to_numpy().reshape(-1,1)
y_test = test_df['y_test'].to_numpy().reshape(-1,1)


print("Square error of Linear regression: ")
print("Square error of SVM regresssion model: ")


x_train = pd.read_csv("../HW2/x_train.csv").values
y_train = pd.read_csv("../HW2/y_train.csv").values[:,0]
x_test = pd.read_csv("../HW2/x_test.csv").values
y_test = pd.read_csv("../HW2/y_test.csv").values[:,0]


print("Accuracy of FLD: ")
print("Accuracy SVM classification model: ")



x_train = pd.read_csv("../HW3/x_train.csv")
y_train = pd.read_csv("../HW3/y_train.csv")
x_test = pd.read_csv("../HW3/x_test.csv")
y_test = pd.read_csv("../HW3/y_test.csv")


print("Accuracy of Decision Tree: ")
print("Accuracy of Random Forest: ")
print("Accuracy SVM classification model: ")


