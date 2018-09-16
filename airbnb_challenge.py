import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import CategoricalEncoder, RobustScaler, StandardScaler
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

DATA_FILE = 'TH_data_challenge.tsv'
rcParams['figure.figsize'] = 12, 4

def convert_categorical_features(df):
    enc = CategoricalEncoder(encoding='ordinal')
    encoded_features = enc.fit_transform(
            df[['dim_is_requested',
                'dim_market',
                'dim_room_type',
                'cancel_policy',
                'dim_is_instant_bookable']])
   
    encoded_df = pd.DataFrame(
            encoded_features,
            index=df.index,
            columns=[
                'dim_is_requested',
                'dim_market',
                'dim_room_type',
                'cancel_policy',
                'dim_is_instant_bookable'])
   
    col = df.columns.tolist()
    col_non_cat = col[1:3] + col[5:6] + col[7:10] + col[11:]
    df_non_cat = df[col_non_cat]
    col_cat = encoded_df.columns.tolist()
    col_full = col_cat[:] + col_non_cat[:]

    stack_full = np.column_stack([encoded_df, df_non_cat])
    stack_df = pd.DataFrame(stack_full, index=df.index, columns=col_full)
    return stack_df


def split_dataset(ratio = 0.2):
    df = pd.read_csv(DATA_FILE, sep='\t')
    
    #Dedupe
    df.duplicated(keep='first').sum()
    df = df.drop_duplicates(keep='first')

    #Remove outliers
    df_no_outliers = df[(df.m_effective_daily_price > 0)]

    #Drop unused columns
    df_select = df_no_outliers.drop(columns=['ds_night',
                                           'ds',
                                           'id_listing_anon',
                                           'id_user_anon',
                                           'dim_lat',
                                           'dim_lng'])
    
    stack_df = convert_categorical_features(df_select)

    y = stack_df['dim_is_requested']
    X = stack_df.drop(columns='dim_is_requested')

    train_inputs, test_inputs, train_output, test_output = train_test_split(
            X, y, test_size=ratio, random_state=42)

    return train_inputs, test_inputs, train_output, test_output

def modelfit(alg, X_train, y_train, X_test, y_test, 
        useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(
                xgb_param, 
                xgtrain, 
                num_boost_round=alg.get_params()['n_estimators'], 
                nfold=cv_folds,
                metrics='auc', 
                early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Predict test set:
    dtest_predictions = alg.predict(X_test)
    dtest_predprob = alg.predict_proba(X_test)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy (Train): %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    print("Accuracy (Test): %.4g" % metrics.accuracy_score(y_test, dtest_predictions))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, dtest_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
param_test1 = {
        'max_depth': [4,7,10],
        'min_child_weight': [1,4,7]}

param_test2 = {
        'gamma':[i/10.0 for i in range(0,10,2)]
        }

param_test3 = {
        'subsample':[i/10.0 for i in range(6,11,2)],
        'colsample_bytree':[i/10.0 for i in range(6,11,2)] }

param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

param_test5 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}

def tune_parameters(param, baseline, X, y):
    gsearch = GridSearchCV(
            estimator=baseline,
            param_grid=param,
            scoring='roc_auc',
            n_jobs=6,
            iid=False,
            cv=5)
    gsearch.fit(X, y)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)

if __name__ == '__main__':
    train_inputs, test_inputs, train_output, test_output = split_dataset()
    xgb1 = XGBClassifier(
            learning_rate=0.1,
            n_estimators=5000,
            max_depth=10,
            min_child_weight=1,
            gamma=0.4,
            reg_lambda=1e-05,
            subsample=1.0,
            colsample_bytree=0.6,
            objective= 'binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
    tune_parameters(param_test5, xgb1, train_inputs, train_output)
    modelfit(xgb1, train_inputs, train_output, test_inputs, test_output)
