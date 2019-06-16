## TASK
================================
1. Partition your data into a holdout set and 5 stratified CV folds.
2. Pick any two machine learning algorithms from the list below, and build a binary
classification model with each of them:
   * Regularized Logistic Regression (scikit-learn)  
   * Gradient Boosting Machine (scikit-learn, XGBoost or LightGBM)  
   * Neural Network (Keras), with the architecture of your choice  
3. Both of your models must make use of numeric, categorical, text, and date features.
4. Compute out-of-sample LogLoss and F1 scores on cross-validation and holdout.
5. Which one of your two models would you recommend to deploy? Explain your decision.
6. (Advanced, optional) Which 3 features are the most impactful for your model? Explain
your methodology.


## ANSWER
================================  
Author: Wandi Susanto  

**Code usage:**
use python 3.6.5  
execute python source\predict_loan.py  

1. I divide dataset to a holdout set (70/30) and 5 CV

2. Choosen algorithms  
A. Regularized Logistic Regression (scikit-learn): with l1 regularizer  
B. Gradient Boosting Machine (scikit-learn)  

3. Use numberic, categorical, text, and date features  
I use all feature data type. Followings are preprocessing and data engineering for each of data type:  
On **numeric data**, to handle empty data I apply mean **imputation** and add binary feature for missing data. There are more complex techniques to estimate the missing value, e.g. using KNN or other classifiers, MICE.  
On **categorical data**, I apply **one-hot encoding**.  
On **date data**, I **feature engineer** date data to data like year, month, day of week, etc.  
On **text data**, I use **feature hashing** for speed and efficient memory consumption.  

   Since the data is very **imbalance**, I apply **SMOTE oversampling** (only on TRAIN data) to make balance data. This significantly improves the model's performance, which can be seen on the train set performance (report_featselect.txt vs report_featselect_no_res.txt).
   I also do **standarization** (zero mean, unit variance or standard deviation = 1) on train data and then apply to test set. It improves performance compared to with standarization.

4. Compute out-of-sample LogLoss and F1 scores on cross-validation and holdout.  
   Done.  
   Please check performance and evaluations in **report_featselect.txt**.  

5. Choose 1 from 2 models and explain  
I choose **Gradient Boosting (GB)** over **Logistic Regression (LR)**. Based on evaluation with given metrics, GradientBoosting shows more robust performance.  

   Based on cross validation (cv) and test evaluation on f1-score and LogLoss metrics:  
   LR has high bias (underfit) problem. Train error relatively high (LogLoss and f1 score low), average cv test error similar to average train error.  
   GB has low bias and slightly high variance (overfit). It has very low average cv test error. Its cv test error is bigger than train error.  
   GB model can fit the data much better than LR. GB and LR perform comparably poorly on test set. Both models suffer from high variance in the data, perform well on training set but poorly on test set.  

   To improve: increase regularization terms, get more data, use model that has low variance which is more robust to data changes, e.g. random forest.  

6. Three Most important features  
   a. annual_inc  
   b. collection_12_mths_ex_med  
   c. deb_to_income  

   explanation:  
   I use ExtraTreeClassifier (scikit-learn), an extremely randomized tree classifier. The ensemble trees compute relative importance of each feature using Gini Importance or Mean Decrease in Impurity (MDI). It calculates each feature importance as total decrease in node impurity averaged over all trees of the ensemble. These importance values inform how significant/important the features and can be used for feature selection.   

