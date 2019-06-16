import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tempfile import TemporaryFile
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

print(sys.version) #use 3.6.5

def overview_data(X):
    """Overview data."""
    # print(X.head)
    print(f"{X.dtypes}\n")
    print(f"Sum of null values in each feature:\n{35 * '-'}")
    X_null = X.isnull().sum()
    print(f"{X_null}")

def plot_boxplot(df, column):
    """Helper function to plot boxplot"""
    boxplot = df.boxplot(column=feat_numeric, return_type='axes')
    plt.savefig("boxplot.png")

def check_data_balance(df, target):
    """Get number of positive and negative examples"""
    pos = df[df[target] == 1].shape[0]
    neg = df[df[target] == 0].shape[0]
    print(f"Positive = {pos}")
    print(f"Negative = {neg}")
    print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")
    plt.figure(figsize=(8, 6))
    sns.countplot(df[target])
    plt.xticks((0, 1), ["is_good", "is_bad"])
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title("Class counts", y=1, fontdict={"fontsize": 20})
    #plt.savefig(out_dir+'\\data_balance.png')

def write_report(file, txt, mode):
    """Helper to write report"""
    f = open(file, mode)
    f.write(txt)
    f.close()

def onehotenc(df, in_feature_names):
    """Apply onehotencoder on category features"""
    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    X = enc.transform(df).toarray()
    out_features_names = enc.get_feature_names(in_feature_names)
    #print("categories: ", enc.categories_)
    #print("feature names: ", out_features_names)
    return X, out_features_names

def impute_data(X, feature_name_in):
    """Impute numeric data"""
    to_replace_dict = {'na':np.nan}
    for i in feature_name_in:
        na_cnt = 0
        if pd.api.types.is_string_dtype(X[i]):
            na_cnt = X[i].str.contains('na').sum()
        if na_cnt > 0:
            X[i] = X.replace(to_replace=to_replace_dict, value=None)

    indicator = MissingIndicator(error_on_new=True, features='all', missing_values=np.nan, sparse=False)
    X_binary_miss = indicator.fit_transform(X).astype(int)
    X_binary_miss_sum = np.sum(X_binary_miss, axis=0)
    feature_name_out = feature_name_in.copy()
    to_del = []
    for i in range(0, len(X_binary_miss_sum)):
        if X_binary_miss_sum[i] > 0:
            feature_name_out.append(feature_name_in[i]+"_miss")
        else:
            to_del.append(i)
    X_binary_miss = np.delete(X_binary_miss, to_del, axis=1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X_tr = imp.transform(X)
    X_out = np.concatenate((X_tr, X_binary_miss), axis=1)
    #print(feature_name_out)
    #print(X_out)
    return X_out, feature_name_out

def engineer_date(df, feat_date):
    """Feature engineer date data"""
    date_format = "%m/%d/%Y"
    X_date = np.zeros((df.shape[0], 5))
    X_date[:,0] = df[feat_date].map(lambda x: pd.to_datetime(x, format=date_format, errors='ignore').year)
    X_date[:,1] = df[feat_date].map(lambda x: pd.to_datetime(x, format=date_format, errors='ignore').month)
    X_date[:,2] = df[feat_date].map(lambda x: pd.to_datetime(x, format=date_format, errors='ignore').days_in_month)
    X_date[:,3] = df[feat_date].map(lambda x: pd.to_datetime(x, format=date_format, errors='ignore').dayofweek)
    X_date[:,4] = df[feat_date].map(lambda x: pd.to_datetime(x, format=date_format, errors='ignore').weekofyear)
    out_feature_names = ["year", "month", "days_in_month", "dayofweek", "weekofyear"]

    X_date_imp, feature_date_imp = impute_data(pd.DataFrame(X_date, columns=out_feature_names), out_feature_names)

    return X_date_imp, feature_date_imp

def vectorize_text_tfidf(X_in, feat_text_in):
    """tfidf vectorization on text data"""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_in)
    # encode text
    X_out = np.empty((0, vectorizer.transform([X_in[0]]).shape[1]))
    for i in range(0,len(X_in)):
        vector = vectorizer.transform([X_in[i]]).toarray()
        X_out = np.append(X_out, vector, axis=0)
        #print(X_out)
    X_out = np.array([x for x in X_out])
    feat_text_out = []
    for n in range(0,X_out.shape[1]):
        feat_text_out.append(feat_text_in+"_"+ str(n))

    return X_out, feat_text_out

def vectorize_text_hashing(X_in, feat_text_in):
    """Hashing vectorization"""
    #better speed and memory consumption with hashing
    vectorizer = HashingVectorizer(n_features= 10)
    X_out = np.empty((0, vectorizer.transform([X_in[0]]).shape[1]))
    for i in range(0,len(X_in)):
        vector = vectorizer.transform([X_in[i]]).toarray()
        X_out = np.append(X_out, vector, axis=0)
    X_out = np.array([x for x in X_out])
    feat_text_out = []
    for n in range(0,X_out.shape[1]):
        feat_text_out.append(feat_text_in+"_"+ str(n))

    return X_out, feat_text_out


def vectorize_all_text(X_in, feat_text_in):
    """Vectorize all text feature column"""
    #print(X_in)
    X_in = X_in.fillna(value="")
    feat_text_out = []
    X_out = np.empty((X_in.shape[0], 0))
    for i in feat_text_in:
        X_tmp, feat_tmp = vectorize_text_hashing(X_in[i], i)
        X_out = np.append(X_out, X_tmp, axis=1)
        feat_text_out.extend(feat_tmp)

    #print(X_out.shape)
    return X_out, feat_text_out


def preproc_data(X, feat_list):
    """Function to preprocessing all type of data: data imputation, onehotencoding, date feature engineering, text vectorization."""
    #numeric data imputation
    X_numeric, feature_numeric = impute_data(X[feat_list['feat_numeric']], feat_list['feat_numeric'])

    #apply one hot encoding to categorical data
    X_category, feature_names_category = onehotenc(X[feat_list['feat_category']], feat_list['feat_category'])
    #feature engineer date data
    X_date, feature_names_date = engineer_date(X[[feat_list['feat_date']]], feat_list['feat_date'])

    #vectorize text data
    X_text, feature_names_text = vectorize_all_text(X[feat_list['feat_text']], feat_list['feat_text'])

    #dump matrix to save time
    #out_dir = r'out'
    #outfile_1 = out_dir + r'\\x_text.npy'
    #outfile_2 = out_dir + r'\\feature_names_tex.npy'
    #np.save(outfile_1, X_text)
    #np.save(outfile_2, feature_names_text)

    #X_text= np.load(outfile_1)
    #feature_names_text= np.load(outfile_2)

    #concatenate all data
    X = np.concatenate((X_numeric, X_category, X_date, X_text), axis=1)
    feature_names = feature_numeric.copy()
    feature_names.extend(feature_names_category)
    feature_names.extend(feature_names_date)
    feature_names.extend(feature_names_text)
    print("all X shape: ", X.shape)
    print("all feature_name: ", len(feature_names))
    return X, feature_names


def eval_metrics(y_true, y_pred, y_score, msg=""):
    """Evaluation metrics."""
    print(msg)
    #requested metrics log_loss and f1 score
    print("\nEvaluation ", msg , file=open('report.txt','a'))
    print("  log loss score (requested metrics): ", metrics.log_loss(y_true, y_score), file=open('report.txt','a'))
    print("  f1 score (requested metrics): ", metrics.f1_score(y_true, y_pred), file=open('report.txt','a'))
    #print("  roc-auc: ", metrics.roc_auc_score(y_true, y_score), file=open('report.txt','a'))
    print("  confusion matrix: \n", metrics.confusion_matrix(y_true, y_pred))
    print("  precision: ", metrics.precision_score(y_true, y_pred), file=open('report.txt','a'))
    print("  recall: ", metrics.recall_score(y_true, y_pred), file=open('report.txt','a'))


def eval_test(clf, X_test, y_test, X_train, y_train, estimator_name):
    """Evaluation on test set."""
    print(f"{42*'='}")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,0]
    #print(f"y_pred {y_pred}"")
    #print(f"y_prob {y_prob}")
    eval_metrics(y_test, y_pred, y_prob, estimator_name+" performance on test set:")
    print(f"{estimator_name} mean acc on test data: {clf.score(X_test, y_test)}")
    print(f"{42*'-'}")
    print("Compute on train set for comparison:")
    y_train_pred = clf.predict(X_train)
    y_train_prob = clf.predict_proba(X_train)[:,0]
    eval_metrics(y_train, y_train_pred, y_train_prob, estimator_name+" performance on train set (for comparison):")
    print(f"{estimator_name} mean acc on train data: {clf.score(X_train, y_train)}")
    print(f"{42*'='}")

def do_cv(clf, X_train, y_train, k, num_job, estimator_name):
    """Perform cross validation"""
    score_eval = ['neg_log_loss', 'f1'] #requested metrics
    #stratified CV
    cv_result = cross_validate(clf, X_train, y_train, scoring=score_eval, cv=k,
                return_train_score=True, n_jobs=num_job)
    print(cv_result)
    mean_test_neg_log_loss = np.mean(cv_result['test_neg_log_loss'])
    mean_test_f1 = np.mean(cv_result['test_f1'])
    mean_train_neg_log_loss = np.mean(cv_result['train_neg_log_loss'])
    mean_train_f1 = np.mean(cv_result['train_f1'])
    txt_1 = "\n  Average test_neg_log_loss: "+str(mean_test_neg_log_loss)+"\n  Average test_f1: "+str(mean_test_f1)
    txt_2 = "\n  Average train_neg_log_loss: " + str(mean_train_neg_log_loss)+"\n  Average train_f1: " + str(mean_train_f1)
    txt = "\nCross validation " + estimator_name + txt_1 + txt_2
    print(txt)
    write_report("report.txt", txt, "a")


def feature_selection(X, y, feature_names, do_apply_feature_selection):
    """Feature selection and get most important features"""
    print("X.shape before feature selection: ", X.shape)
    X = pd.DataFrame(X, columns=feature_names)
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    print("Feature ranking:", file=open('report.txt','a'))
    for f in range(X.shape[1]):
        print(f"  {f + 1}. feature {indices[f]} {X.columns.values[f]} {importance[indices[f]]}", file=open('report.txt','a'))
        if f==9:
            break

    model = SelectFromModel(clf, prefit=True)
    return model


