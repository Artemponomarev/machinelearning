# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from feature_selector import FeatureSelector



# Importing the dataset
dataset1 = pd.read_excel('Data_set.xlsx',nrows=97)
train_labels = dataset1.iloc[:, 6].values
dataset = dataset1.iloc[:, [15,113,16,110,18,19,40,69,82,105,21,109,13,20,112,108,12,66,79,5,50,85,173,174,162,161,154,156,206,219,214,222,233,230]]
dataset = dataset.fillna(0)
fs = FeatureSelector(data = dataset, labels= train_labels)

fs.identify_zero_importance(task = 'regression', eval_metric = 'l2', n_iterations = 100, early_stopping = True)
zero_importance_features = fs.ops['zero_importance']
fs.identify_low_importance(cumulative_importance = 0.5)
low_importance_features = fs.ops['low_importance']
fs.identify_collinear(correlation_threshold = 0.9)
fs.plot_collinear(plot_all = True)
correlated_features = fs.ops["collinear"]
fs.record_collinear.head()
dataset_with_goog_features = fs.remove(methods = 'all')




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_with_goog_features, train_labels, test_size = 0.75)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = "liblinear")
classifier.fit(X_train, y_train)


print("b0: ",classifier.intercept_)
print("b1...bn array: ",classifier.coef_)




# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

