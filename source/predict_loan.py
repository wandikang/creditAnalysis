import sys
import numpy as np
from source.predict_helper import *

print(sys.version) #use 3.6.5

if __name__ == "__main__":
    #load dataset, set main directory
    main_dir = r'C:\\...\\Documents\\playg\\predictCredit'
    dataset_lc = main_dir + r'\\data\\DR_Demo_Lending_Club.csv'
    out_dir = main_dir + "\\out"
    print("\n====START===", file=open("report.txt", "a"))

    #simple boolean configs
    do_get_important_features  = 1  #set to 0 or 1
    do_apply_feature_selection = 1  #do_get_important_features must be 1
    do_cv_A    = 1  #cv on Logistic Regression
    do_model_A = 1  #Logistic Regression eval on test set
    do_cv_B    = 1  #cv on Gradient Boosting
    do_model_B = 1  #Gradient Boosting eval on test set
    num_k_cv = 5 #num k for cross validation
    num_job = 2 #num parallel job

    df = pd.read_csv(dataset_lc)
    print(f"{42*'='}")
    overview_data(df)
    print(f"{42*'='}")
    target = "is_bad"
    check_data_balance(df, target)
    print(f"{42*'='}")

    feat_numeric = ["annual_inc", "collections_12_mths_ex_med", "debt_to_income", "delinq_2yrs", "emp_length",
    "Id", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_major_derog", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
    feat_category = ["addr_state", "home_ownership", "initial_list_status", "pymnt_plan", "policy_code",
    "purpose_cat", "verification_status", "zip_code"]
    feat_text = ["emp_title", "Notes", "purpose"]
    feat_date = "earliest_cr_line"

    feat_list = {'feat_numeric': feat_numeric, 'feat_category': feat_category,
                'feat_date': feat_date, 'feat_text': feat_text}

    #remove target
    X = df.drop([target], axis=1)
    y = df[target]
    #preprocessing: data imputation, onehotencoding, date feature engineering, text vectorization
    X, feature_names = preproc_data(X, feat_list)

    #split train-test data
    print("Split train-test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 77)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train pos", np.sum(y_train))
    print("y_test pos", np.sum(y_test))

    #feature selection
    if do_get_important_features == 1:
        print("get important features")
        model_feat_select = feature_selection(X_train, y_train, feature_names, do_apply_feature_selection)
        if do_apply_feature_selection == 1:
            print("apply feature selection")
            print("apply feature selection", file=open("report.txt", 'a'))
            X_train = model_feat_select.transform(X_train)
            print("X.shape after feature selection: ", X_train.shape)
            X_test = model_feat_select.transform(X_test) #apply feature selection

    #handle imbalance dataset
    print("resampling training samples ...")
    #undersampling
    #cc = ClusterCentroids(sampling_strategy=1.0, random_state=8)
    #X_tr_res, y_tr_res = cc.fit_resample(X, y)
    #SMOTE oversampling
    sm = SMOTE(sampling_strategy=1.0, random_state=32)
    X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)

    pos = np.sum(y_tr_res)
    neg = len(y_tr_res) - pos
    print("total samples: ", len(y_tr_res))
    print("positive samples: ", pos) #should be half to train set after resampling
    print(f"proportion positive samples to negative samples after resampling: {(pos/neg)*100:.2f}%")

    #normalization
    #X_tr_res = preprocessing.normalize(X_tr_res, norm='l2')
    #X_test = preprocessing.normalize(X_test, norm='l2')

    #standarize data - zero mean and unit variance/standard deviation of one
    #standarization gives better result than normalization, LogReg converges faster
    print("standardize data")
    scaler = preprocessing.StandardScaler().fit(X_tr_res)
    X_tr_res = scaler.transform(X_tr_res)
    X_test = scaler.transform(X_test) #apply standarization

    #feat_numeric = ["annual_inc", "debt_to_income"]
    #plot_boxplot(X_tr_res, feat_numeric)
    #exit()

    #model A is for LogisticRegression, model B is for GradientBoosting
    regularizer = 'l1'
    #cross validation
    print("number cv k-fold = 5", file=open("report.txt", 'a'))
    if do_cv_A:
        print("cross validate LogisticRegression... ")
        clf_A = LogisticRegression(solver='liblinear', penalty=regularizer, max_iter=100, verbose=0)
        do_cv(clf_A, X_tr_res, y_tr_res, num_k_cv, num_job, "LogisticRegression")

    if do_cv_B:
        print("cross validate GradientBoosting ... ")
        clf_B = ensemble.GradientBoostingClassifier(n_estimators= 500, max_leaf_nodes=8, max_depth=6,
            random_state= 22, min_samples_split=10, learning_rate=0.1, subsample=0.5)
        do_cv(clf_B, X_tr_res, y_tr_res, num_k_cv, num_job, "GradientBoosting")

    # evaluation on test set
    # model A
    if do_model_A:
        print(f"{42*'='}")
        print("eval Logistic Regression on test set")
        clf_A = LogisticRegression(solver='liblinear', penalty=regularizer, max_iter=100, verbose=0).fit(X_tr_res, y_tr_res)
        eval_test(clf_A, X_test, y_test, X_tr_res, y_tr_res, "LogisticRegression")

    #model B
    if do_model_B:
        print(f"{42*'='}")
        print("eval GradientBoosting on test set")
        clf_B = ensemble.GradientBoostingClassifier(n_estimators= 500, max_leaf_nodes=8, max_depth=5,
                random_state= 22, min_samples_split=10, learning_rate=0.1, subsample=0.5)
        clf_B.fit(X_tr_res, y_tr_res)
        eval_test(clf_B, X_test, y_test, X_tr_res, y_tr_res, "GradientBoosting")

    print("\n=== end ===\n", file=open("report.txt", "a"))

