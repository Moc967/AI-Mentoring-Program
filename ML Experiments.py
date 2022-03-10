# -*- coding: utf-8 -*-
"""Machine Learning Experiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y9YGsn6dSWAroDwz134hfgrlxeluX3Nk

# Boston House Price Forecast

The development in this experiment is based on ModelArts. For details about how to set 
up the environment, see the HCIA-AI V3.0 Experiment Environment Setup Guide. The 
sample size of the dataset used in this case is small, and the data comes from the open 
source Boston house price data provided by scikit-learn. The Boston House Price Forecast 
project is a simple regression model, through which you can learn some basic usage of the 
machine learning library sklearn and some basic data processing methods.
"""

#Prevent unnecessary warnings.
import warnings
warnings.filterwarnings("ignore")
#Introduce the basic package of data science.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns
##Set attributes to prevent garbled characters in Chinese.
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#Introduce machine learning, preprocessing, model selection, and evaluation indicators.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
#Import the Boston dataset used this time.
from sklearn.datasets import load_boston
#Introduce algorithms.
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNet
#Compared with SVC, it is the regression form of SVM.
from sklearn.svm import SVR
#Integrate algorithms.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#Load the Boston house price data set.
boston = load_boston()
#x features, and y labels.
x = boston.data
y = boston.target
#Display related attributes.
print('Feature column name')
print(boston.feature_names)
print("Sample data volume: %d, number of features: %d"% x.shape)
print("Target sample data volume: %d"% y.shape[0])

x = pd.DataFrame(boston.data, columns=boston.feature_names)
x.head()

sns.distplot(tuple(y), kde=False, fit=st.norm)

#Segment the data.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)
#Standardize the data set.
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
x_train[0:100]

#Set the model name.
names = ['LinerRegression',
'Ridge',
'Lasso',
'Random Forrest',
'GBDT',
'Support Vector Regression',
'ElasticNet',
'XgBoost']
#Define the model.
# cv is the cross-validation idea here.
models = [LinearRegression(),
RidgeCV(alphas=(0.001,0.1,1),cv=3),
LassoCV(alphas=(0.001,0.1,1),cv=5),
RandomForestRegressor(n_estimators=10),
GradientBoostingRegressor(n_estimators=30),
SVR(),
ElasticNet(alpha=0.001,max_iter=10000),
XGBRegressor()]
# Output the R2 scores of all regression models.
#Define the R2 scoring function.
def R2(model,x_train, x_test, y_train, y_test):
 model_fitted = model.fit(x_train,y_train)
 y_pred = model_fitted.predict(x_test)
 score = r2_score(y_test, y_pred)
 return score
#Traverse all models to score.
for name,model in zip(names,models):
 score = R2(model,x_train, x_test, y_train, y_test)
 print("{}: {:.6f}".format(name,score.mean()))

'''
'kernel': kernel function
 'C': SVR regularization factor
 'gamma': 'rbf', 'poly' and 'sigmoid' kernel function coefficient, which affects the model performance
'''
parameters = {
'kernel': ['linear', 'rbf'],
 'C': [0.1, 0.5,0.9,1,5],
 'gamma': [0.001,0.01,0.1,1]
}
#Use grid search and perform cross validation.
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

print("Optimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)

##Perform visualization.
ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)
#Set the canvas.
plt.figure(figsize=(16,8), facecolor='w')
#Draw with a red solid line.
plt.plot (ln_x_test, y_test, 'r-', lw=2, label=u'Value')
#Draw with a green solid line.
plt.plot (ln_x_test, y_predict, 'g-', lw = 3, label=u'Estimated value of the SVR algorithm, $R^2$=%.3f' % 
(model.best_score_))
#Display in a diagram.
plt.legend(loc ='upper left')
plt.grid(True)
plt.title(u"Boston Housing Price Forecast (SVM)")
plt.xlim(0, 101)
plt.show()

"""gcloud ai-platform jobs stream-logs $JOB_ID"""