# -*- coding: utf-8 -*-
"""Scikit_Learn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yA7A0pgIbMYN-mVsNqOg94l3d4fLRqHJ

<a href="https://colab.research.google.com/github/DanRHowarth/Artificial-Intelligence-Cloud-and-Edge-Implementations
/blob/master/Oxford_scikit_learn.ipynb" target="_parent"><img
src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# `scikit-learn` Tutorial * This notebook covers different code implements from the scikit learn library, including:
* Rescaling and Standardizing Data * Model Evaluation and Metrics * Classification and regression machine learning
algorithms * Regularization * Optimising parameters * Ensemble models * Building Pipelines * We will use
classification and regression datasets from `scikit-learn`, and will take code from the *ands-On Machine Learning
with Scikit-Learn and TensorFlow* coursebook, as well as the [Machine Learning with Python Cookbook](
https://www.amazon.co.uk/Machine-Learning-Python-Cookbook-Chris/dp/1491989386/ref=pd_sim_14_5?_encoding=UTF8&pd_rd_i
=1491989386&pd_rd_r=e03d308c-dd3d-11e8-9244-bbcc5e42676d&pd_rd_w=lQiym&pd_rd_wg=lVZwd&pf_rd_i=desktop-dp-sims&pf_rd_m
=A3P5ROKL5A1OLE&pf_rd_p=1e3b4162-429b-4ea8-80b8-75d978d3d89e&pf_rd_r=KBZG3157A75PS5MPGY8K&pf_rd_s=desktop-dp-sims
&pf_rd_t=40701&psc=1&refRID=KBZG3157A75PS5MPGY8K).





### Approach and Exercises * Note that this notebook is not focussed on how to structure an end to end machine
learning problem, but is about extending the code implementations you will see in the end-to-end tutorials to further
your knowledge. * This notebook is a progression from the previous notebooks, meaning that the exercises will
generally be more involved. We will provide code implementations for one of the datasets and leave space for you to
code implementations on a different dataset. Support is available via Teams. * Each section will have an exercise to
help reinforce your learning. We suggest you: * Write out each line of code by hand (rather than copy and paste it
from the relevant example) - this will improve your understanding of code syntax * Write out, above each line of
code, an explanation as to what the code, using a comment `#` - this will improve your understanding of how the code
works

## 1. Tutorial set up 
* Import statements
* Loading the datasets
* Reviewing the ML problem
"""

# import statements 
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

import pandas as pd
import numpy as np

# Hide all warning messages
import warnings

warnings.filterwarnings('ignore')

# we will generally import scikit learn libraries when they are required, so you understand which specifc libraries
# are required

# load the classification dataset
class_data = load_breast_cancer()
class_X = pd.DataFrame(class_data.data, columns=class_data.feature_names)
class_y = class_data.target

# check the data has loaded successfully 
print(class_X.shape)
print(class_y.shape)

class_X.head()

# load the regression dataset
boston = load_boston()
boston_X = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_y = boston.target

# check the data has loaded successfully 
print(boston_X.shape)
print(boston_y.shape)

"""## 2. Rescaling and Standardizing Data * Machine Learning algorithms can receive performance benefits from data 
that has been rescaled and standardized, either in terms of speed or accuracy, or both. * There are a number of 
different ways of doing this, and implementations may vary because of algorithm requirements or personal preference. 
* We cover two common implementations below, and an additional one in the exercise. We'd encourage further research 
to understand the maths in more detail. """

## rescaling using MinMaxScaler, which converts data to a specified range, usually (0,1) or (-1,1)
# many machine learning algorithms assume data is on the same scale  

# here we import the necessary library 
from sklearn.preprocessing import MinMaxScaler

# Create feature - we will use one feature but you may just pass all the features to rescaler 
feature = class_X['texture error']
print(feature)

# print feature prior to rescaling
print("Features prior to rescaling:\n")
print(feature[0:10])

# we will print the shape out as well 
print("\nFeature shape prior to reshaping:")
print(feature.shape)

## because we have used a pandas column, our data has the wrong shape for the function
# we need a 2D tuple

# this will reshape the data 
feature = feature.values.reshape(-1, 1)

# notice we now have a (samples,feature) tuple not just a (samples,) tuple
print("\nFeature shape after reshaping:")
feature.shape

# we need to instantiate the scaler and pass in the range we want our values to be scaled within 
minmax_scale = MinMaxScaler(feature_range=(0, 1))

# Scale the feature - note the fit_transform function, in the next example we will separate these steps
scaled_feature = minmax_scale.fit_transform(feature)

# lets look at the features now we have rescaled
print("Features after rescaling:\n")
scaled_feature[0:10]

## standardizing rescales features so that they are normally distributed, with a mean of 0 and a std deviation of 1

# import the appropriate library
from sklearn.preprocessing import StandardScaler

# we will transform all the features in the dataset, so let's look at it  prior to transformation
mean_std = pd.DataFrame(data={'mean': class_X.mean(), 'std': class_X.std()})

# call the dataframe
mean_std[0:10]

# Create scaler
scaler = StandardScaler()

# fit the scaler - this calculates the minimum and maximum values of the data
scaler.fit(class_X)

# transform the data using the fitted scaler - this applies the transform using the fit
standardized = scaler.transform(class_X)

# this shows the features transformed - note that this returns an array not a dataframe
standardized[0:10]

# lets have another look at our transformed data
mean_std_transformed = pd.DataFrame(data={'mean': standardized.mean(axis=0), 'std': standardized.std(axis=0)},
                                    index=class_X.columns)

# call the dataframe
mean_std_transformed[0:10]

"""### EXERCISE SECTION 2: * Apply the `Rescaling` and `Standardization` functions to one or more features of the 
`Boston` dataset * Research and implement a `Normalizer` function * Have a think about what circumstances each of 
these functions might be most appropriate to apply to different data and why * We can discuss the implementations and 
benefits of each approach on Teams """

## EXERCISE CODE GOES HERE

"""## 3. Model Evaluation and Metrics 
* We need to understand how we are going to assess our model's performance. This involves:
  * Splitting our Training and Testing data appropriately 
  * Choosing an Evaluation Metric
  * Developing a Baseline Model 

"""

## testing and training splits 

# import statement 
from sklearn.model_selection import train_test_split

# this splits our data into a test and training set
# the function shuffles our data before splitting (so we will get both classes in both sets)
# setting a random_state means that this shuffling will be consistent for each run 
# setting stratify=class_y tells the function to have an even number of class labels in each set
X_train, X_test, y_train, y_test = train_test_split(class_X, class_y, test_size=0.3, random_state=1, stratify=class_y)

# we can verify the stratifications using np.bincount
print('Labels counts in y:', np.bincount(class_y))
print('Percentage of class zeroes in class_y', np.round(np.bincount(class_y)[0] / len(class_y) * 100))

print("\n")
print('Labels counts in y_train:', np.bincount(y_train))
print('Percentage of class zeroes in y_train', np.round(np.bincount(y_train)[0] / len(y_train) * 100))

print("\n")
print('Labels counts in y_test:', np.bincount(y_test))
print('Percentage of class zeroes in y_test', np.round(np.bincount(y_test)[0] / len(y_test) * 100))

"""* Note that there are additional ways to divide our data into test and training sets, and we cover `K-fold cross 
validation` in the **Pipeline** section below. """

## we can create a baseline model to benchmark our other estimators against
## this can be a simple estimator or we can use a dummy estimator to make predictions in a random manner 

# this is the required import statement 
from sklearn.dummy import DummyClassifier

# this creates our dummy classifier, and the value we pass in to the strategy parameter determines how the 
# classifier will make the predictions
dummy = DummyClassifier(strategy='uniform', random_state=1)

# "Train" model
dummy.fit(X_train, y_train)

## evaluating the model metrics

# Here we get an accuracy score
dummy.score(X_test, y_test)

"""* There are a number of different ways in `scikit-learn` to get an estimator score and it can get confusing first. 
* Remember that to get a score, we need to instantiate a model, fit it to the data, predict using unseen data, 
compare the predictions against actual data, and score the difference. This is true for classification and regression 
problems, and is true no matter the method used to get there. * So, in the end-to-end tutorials we split the training 
and test data,  fitted our data to an estimator, and called the `.predict` method on the estimator to get our 
predictions, and then passed this to a scoring function (four steps) * In using the `estimator.score()`method above, 
we are passing in our split data and the method is then making predictions and returning the score (three steps). * 
And, in the `cross_val_score()` method used below we are effectively using one step as the method takes an estimator 
and our data and returns a score. You can find out more about this method [here](
https://scikit-learn.org/stable/modules/cross_validation.html) * Another function cross validation function 
`cross_validate()` is available and enables the user to specify multiple scoring metrics at once. """

## here we fit a new estimator and use cross_val_score to get a score based on a defined metric 

# import statements
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# instantiate logistic regression classifier
logistic = LogisticRegression(solver='liblinear')

# we pass our estimator and data to the method. we also specify the number of folds (default is 3)
# the default scring method is the one associated with the estimator we pass in
# we can use the scoring parameter to pass in different scoring methods. Here we use recall.  
cross_val_score(logistic, class_X, class_y, cv=5, scoring="recall")

"""### EXERCISE SECTION 3: * Part 1: Implement binary classification scoring functions * Use `cross_val_score` to 
implement `f1`, and `precision` scores * Understand what these scores, and the `accuracy` and `recall` scores 
implemented above, tell us * Part 2: Another way of assessing our model's performance is with the `Receiving 
Operating Characteristic (ROC) Curve`. * What does this show us? Can you implement it? * Like the other questions 
here, we can work through these via Teams. * Part 3: What are the main ways of evaluating a mutliclass classification 
problem? * Research and make notes below * Part 4: Apply the above approach to the Boston dataset. You should think 
about: * What sort of test and training split you want to implement * Be careful to change the variable names for 
`X_train`, `y_train` etc. so that the new dataset doesn't overwrite those variables * Experiment with not using the 
`stratify =  ` variable * The evaluation metrics you wish to use and why (this is different than above as the Boston 
dataset is a regression problem) * Evaluation metrics are covered in *Python Machine Learning*, Chapter 6 * 
Implementing a baseline model * *Python Machine Learning*, Chapter 10 has linear regression implementations to draw 
from. * Feel free to discuss your approach on Teams and ask for support """

## EXERCISE SECTION 3: PART 1
## use the cross_val_score function to pass in and return the scoring functions: precision, f1 
## note the scores and research what each score represents

## EXERCISE SECTION 3: PART 2
## look up and implement the ROC metric

"""**EXERCISE SECTION 3: PART 3**
* What are the evaluation metrics that can be used for mutliclass regression problems?


---

**Answer**
* Text here

"""

## EXERCISE SECTION 3: PART 4 
## Implement a baseline model and evaluation metric for the Boston dataset (regression)

"""## 4. Classification and regression machine learning algorithms

### EXERCISE SECTION 4: * You already have an idea about how to fit and instatiate machine learning models. * There 
are a large number of machine learning approaches you can use, and this exercise is about understanding how a number 
of them perform on our datasets. * Research and instantiate as many algorithms as you wish in the code cell marked 
below. * There are a number of examples in *Python Machine Learning*, for example `SVM`, `Decision Trees` and 
`Logistic Regression` in Chapter 3. """

## EXERCISE CODE HERE

"""## 5. Regularization 
* When we train our models, we might need to prevent against **over-** or **underfitting**. Overfitting is when our model performs well on the training data but does not generalise to unseen data (such as our test set). The model might be too complex, with the parameter weights probably too large and aligned too closely to the patterns of the training data. The solution is to develop a means of reducing these weights during training, and this is done through a process called **regularization**.
* Underfitting is when our model is too simple to capture patterns in either the training or test sets. Chapters 3 and 10 of *Python Machine Learning* cover this topic in more detail.
* How we regularise the model depends on what estimator we are using. In most cases, we can penalise the increase in parameter weights as the model trains by inserting a regularization term to our model when we instantiate it. This is applicable to regression models, logistic regression and support vector classifiers (see below for implementations).  
* For these estimators, the regularization term is commonly either **L1 Regularization**, where the weights are reduced by a user defined parameter multiplied by the absolute value of the weights (or coefficients); or **L2 Regularization**, where the weights are reduced by a user defined parameter multiplied by the squared sum of the weights.
* For tree based models, such as Decision Trees and Random Forests, we can help prevent overfitting by adjusting parameters associated with the leaf and branch settings.
* Regularization is another reason why feature scaling such as standardization is important. For regularization to work properly, we need to ensure that all our features are on comparable scales.
* We implement L1 and L2 Regularization for regression models in this section. In Section 6, we implement  `RandomSearchCV` (explained below) on a `Logistic Regression` estimator. The parameters that we change in that implementation are related to regularization ('`C`' and '`penalty`').
"""

## For regularization in regression we instantiate new models
## Lasso regression is used for L1 Regularization 

# import statement 
from sklearn.linear_model import Lasso

# Create lasso regression with alpha value - this is our hyperparameter 
regression = Lasso(alpha=0.5)

# Fit the linear regression
model_l1 = regression.fit(class_X, class_y)

## Ridge regression is used for L2 Regularization

# import statement 
from sklearn.linear_model import Ridge

# Create ridge regression with an alpha value (our hyperparameter)
regression = Ridge(alpha=0.5)

# Fit the linear regression
model_l2 = regression.fit(class_X, class_y)

"""### EXERCISE FOR SECTION 5
* Part 1: Train both the models above and get a score. Compare the results and see if one type of regularization works better than another.
* Part 2: The `ElasticNet`estimator combines both L1 and L2 Regularization. Look up the estimator on the scikit-learn documentation and a) write out how it combines both regularization methods, and b) see if you can implement it.
* Part 3: We can use `RidgeCV` and `LassoCV` to explore the best regularization parameters for each estimator. Implement one or both of these estimators, defining the parameters you want to search over.
* Part 4: Look at the `DecisionTreeClassifier `documentation in `scikit-learn` and come to a view on which parameters can be set to help prevent overfitting (and therefore help regularization). 
"""

## EXERCISE SECTION 5: PART 1

## EXERCISE SECTION 5: PART 2

## EXERCISE SECTION 5: PART 3

"""#### EXERCISE SECTION 5: PART 4
* Answer here

## 6. Optimizing parameters 
 * We applied `GridSearchCV` to identify the best hyperparameters for our models in the in-class regression tutorials.
 * There are other methods available to use that don't take the brute force approach of `GridSearchCV`.
 * We will cover an implementation of `RandomizedSearchCV` below, and use the exercise for you to implement it on the other datatset.
  * We use this method to search over defined hyperparameters, like `GridSearchCV`, however a fixed number of parameters are sampled, as defined by `n_iter` parameter.
"""

# import libraries - note we use scipy for generating a uniform distribution 
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# Create logistic regression
logistic = LogisticRegression(solver='liblinear')

# we can create hyperparameters as a list, as in a type regularisation penalty 
penalty = ['l1', 'l2']

# or as a distribution of values to sample from -'C' is the hyperparameter controlling the size of the regularisation penalty 
C = uniform(loc=0, scale=4)

# we need to pass these parameters as a dictionary of {param_name: values}
hyperparameters = dict(C=C, penalty=penalty)

# we instantiate our model
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)

# and fit it to the data 
best_model = randomizedsearch.fit(class_X, class_y)

# and we can call this method to return the best parameters the search returned
best_model.best_estimator_

"""### EXERCISE FOR SECTION 6: 
* Part 1: Once we have fit the model to our data, we can call different methods on it, for example we can find out what the best parameters were, and we can predict using the best estimator returned by the search. Use the docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to call these methods on the `best_model` variable above. 
* Part 2: Implement `RandomSearchCV` on the Boston dataset
  * Think about which parameters you want to specify
  * You can pass these parameters to variables prior to creating the random search (as we do above)
  * Then call methods as we do for Part 1.
"""

## EXERCISE SECTION 6: PART 1

## EXERCISE SECTION 6: PART 2

"""## 7. Ensemble models 

* The goal of ensemble methods is to combine different classifiers into a meta-classifier that has better generalization performance than each individual classifier alone. 
* There are several different approaches to achieve this, including **majority voting** ensemble methods, with which we select the class label that has been predicted by the majority of classifiers.
* The ensemble can be built from different classification algorithms, such as decision trees, support vector machines, logistic regression classifiers, and so on. Alternatively, we can also use the same base classification algorithm, fitting different subsets of the training set. 
* Indeed, majority voting will work best if the classifiers used are different from each other and/or trained on different datasets (or subsets of the same data) in order for their errors to be uncorrelated.  Chapter 7 of *Python Machine Learning* has a good discussion of the principles behind Ensemble models. 
"""

# import our classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# lets instantiate individual models 
log_clf = LogisticRegression(solver='liblinear')
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# and an ensemble of them
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                              # here we select hard voting, which returns the majority of the predictions,
                              # not an average of probabilities
                              voting='hard')

# here we can cycle through the individual estimators 
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    # fit them to the training data
    clf.fit(X_train, y_train)

    # get a prediction
    y_pred = clf.predict(X_test)

    # and print the prediction 
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""### EXERCISE FOR SECTION 7: * Part 1: Implement **soft voting**. * If all the classifiers used in the ensemble are 
able to estimate class probabilities (they have a `predict_proba()` method that can be called on the estimator), 
then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the 
individual classifiers. It often achieves higher performance than hard voting because it gives more weight to highly 
confident votes. *  Replace `voting="hard"` with `voting="soft"` and ensure that all your classifiers can estimate 
class probabilities. * If you use the SVC estimator, you need to set `probability = True`, which will make the SVC 
class use cross-validation to estimate class probabilities and add a predict_proba() method. 

* Part 2: Implement Ensemble Learning on the Boston Dataset.

"""

## EXERCISE SECTION 7: PART 1

## EXERCISE SECTION 7: PART 2

"""## 8. Building Pipelines * We can build a pipeline so that many of the steps we have covered to now can be 
implemented in one step, and be applied to the training and testing set without repeating code. * We will cover one 
implementation of a pipeline. The exercise below will link to further information for you to be able implement your 
own pipelines. """

# import the libraries - these are dependent on what we wish to implement in our pipelines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# this is the pipeline class required to implement all pipelines
from sklearn.pipeline import make_pipeline

# here we instantiate the pipeline with our methods
pipe_lr = make_pipeline(StandardScaler(),  # takes an arbitary number of transfomers
                        PCA(n_components=2),
                        LogisticRegression(random_state=1))  # but ends with an estimator

# here we fit our pipeline to the data, so that the steps above are executed on the training set
pipe_lr.fit(X_train, y_train)

# here we call the predict method on our pipeline model against the test set 
y_pred = pipe_lr.predict(X_test)

# and print out the result
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))

"""### EXERCISE FOR SECTION 8: * Implement a pipeline on the classification dataset (the one used in the example 
above). Specifically, implement `K-Fold Cross Validation` as set out in  *Python Machine Learning*, Chapter 6. 
Moreinformation on pipelines is available at this [link](
https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html) blog series 
* Think about: * the code being implemented for `K-Fold Cross Validation` as part of the pipeline. Put comments above 
the code to explain what is happening. * how you want to preprocess the data * whether you want to think about 
reducing the dimensions of the data with something like `Principal Component Analysis` and why * what algorithm you 
want to use * make sure to predict the data """

## EXERCISE CODE HERE

"""## 9. Review * We have covered: * Scaling and Standardizing Data * Model Evaluation and Metrics * Classification 
and regression machine learning algorithms * Regularization * Optimising parameters * Ensemble models * Building 
Pipelines * We have of course not covered everything that it is possible to cover, and there is plenty more to 
research. Please ask questions on Teams or follow-up with your own research. """
